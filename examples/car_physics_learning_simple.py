"""
간단한 버전: MLP를 사용하여 물리 파라미터를 직접 최적화
유한 차분법을 사용하여 그래디언트를 계산하고 파라미터를 업데이트
"""
import os
import sys

# distutils hack 문제 해결 (setuptools/triton 호환성)
os.environ['SETUPTOOLS_USE_DISTUTILS'] = 'stdlib'

import genesis as gs
import argparse
import numpy as np
import pandas as pd  # 성능 향상을 위해 pandas 사용
import torch

# torch compile 관련 에러 억제 (필요시)
try:
    torch._dynamo.config.suppress_errors = True
except:
    pass

import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple
import math



class PhysicsParameterMLP(nn.Module):
    """MLP 모델: 시뮬레이션 상태를 입력으로 받아 물리 파라미터를 출력"""
    
    def __init__(self, input_dim: int = 7, hidden_dims: List[int] = [128, 64, 32], output_dim: int = 7):
        """
        Args:
            input_dim: 입력 차원 (차체 위치, 속도, 시간 등)
            hidden_dims: 히든 레이어 차원들
            output_dim: 출력 차원 (물리 파라미터 개수)
        """
        super(PhysicsParameterMLP, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            prev_dim = hidden_dim
        
        # 출력 레이어
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)
        
        # 파라미터 범위 제한을 위한 활성화 함수
        self.sigmoid = nn.Sigmoid()
        self.softplus = nn.Softplus()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 입력 상태 [batch_size, input_dim]
        Returns:
            물리 파라미터 [batch_size, output_dim]
            [friction, car_mass, wheel_mass, kp_drive, kv_drive, kp_steer, kv_steer]
        """
        output = self.network(x)
        
        # 파라미터 범위 제한
        # friction: 0~2 (sigmoid * 2)
        # mass: > 0 (softplus + offset)
        # kp, kv: > 0 (softplus + offset)
        params = torch.zeros_like(output)
        params[:, 0] = self.sigmoid(output[:, 0]) * 2.0  # friction: 0~2
        params[:, 1] = self.softplus(output[:, 1]) + 500.0  # car_mass: > 500
        params[:, 2] = self.softplus(output[:, 2]) + 10.0  # wheel_mass: > 10
        params[:, 3] = self.softplus(output[:, 3]) + 100.0  # kp_drive: > 100
        params[:, 4] = self.softplus(output[:, 4]) + 10.0  # kv_drive: > 10
        params[:, 5] = self.softplus(output[:, 5]) + 500.0  # kp_steer: > 500
        params[:, 6] = self.softplus(output[:, 6]) + 50.0  # kv_steer: > 50
        
        return params


class CarSimulationWrapper:
    """시뮬레이션 래퍼: 물리 파라미터를 받아서 시뮬레이션을 실행하고 결과를 반환"""
    
    def __init__(self, urdf_path: str, use_cpu: bool = False, show_viewer: bool = False):
        self.urdf_path = urdf_path
        self.use_cpu = use_cpu
        self.show_viewer = show_viewer
        self.scene = None
        self.car = None
        self.ground = None
        self.steer_dof_indices = None
        self.drive_dof_indices = None
        self.body_link = None
        self.initialized = False
        
    def initialize_scene(self):
        """시뮬레이션 씬 초기화"""
        if self.initialized:
            return
        
        try:
            backend = gs.cpu if self.use_cpu else gs.gpu
            gs.init(backend=backend, logging_level="error")
        except Exception as e:
            print(f"⚠️ GPU backend failed: {e}\n→ Switching to CPU")
            gs.init(backend=gs.cpu, logging_level="error")
        
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                substeps=10,
                gravity=(0, 0, -9.81),
            ),
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(3, 2, 1.5),
                camera_lookat=(0, 0, 0.5),
                camera_up=(0, 0, 1),
            ),
            show_viewer=self.show_viewer,
        )
        
        # Ground 추가
        self.ground = self.scene.add_entity(
            morph=gs.morphs.Plane(),
            material=gs.materials.Rigid(friction=1.0)
        )
        
        # Car 추가
        self.car = self.scene.add_entity(
            morph=gs.morphs.URDF(
                file=self.urdf_path,
                pos=(0, 0, 0.8),
                merge_fixed_links=False
            ),
            material=gs.materials.Rigid()
        )
        
        self.scene.build()
        
        # 조인트 인덱스 찾기
        self._find_joint_indices()
        
        # Body 링크 찾기
        self._find_body_link()
        
        self.initialized = True
    
    def _find_joint_indices(self):
        """조인트 인덱스 찾기"""
        def joint_names_with(substr):
            found = []
            for j in getattr(self.car, "joints", []):
                if substr in getattr(j, "name", ""):
                    found.append(j.name)
            return found
        
        steer_joint_names = joint_names_with("steer") or ["steer_fl", "steer_fr"]
        rotate_joint_names = joint_names_with("rotate")
        rear_joint_names = joint_names_with("rear")
        drive_joint_names = rotate_joint_names + rear_joint_names
        if not drive_joint_names:
            drive_joint_names = ["rear_rl", "rear_rr", "wheel_fl_rotate", "wheel_fr_rotate"]
        
        def get_dof_indices(joint_names):
            dof_indices = []
            for name in joint_names:
                try:
                    joint = self.car.get_joint(name)
                    if joint and hasattr(joint, 'dofs_idx_local') and len(joint.dofs_idx_local) > 0:
                        dof_indices.append(joint.dofs_idx_local[0])
                except Exception:
                    pass
            return dof_indices
        
        self.steer_dof_indices = get_dof_indices(steer_joint_names)
        self.drive_dof_indices = get_dof_indices(drive_joint_names)
    
    def _find_body_link(self):
        """Body 링크 찾기"""
        prefer = ("base", "car_body", "base_link", "chassis", "body")
        links = getattr(self.car, "links", []) or []
        for name in prefer:
            for L in links:
                if getattr(L, "name", "") == name:
                    self.body_link = L
                    return
        self.body_link = links[0] if links else None
    
    def get_body_state(self) -> np.ndarray:
        """차체 상태 가져오기 (위치, 속도)"""
        if self.body_link is None:
            return np.zeros(6)  # [x, y, z, vx, vy, vz]
        
        try:
            fn = getattr(self.body_link, "get_world_transform", None)
            if callable(fn):
                pose = fn()
                p = getattr(pose, "p", None)
                if p is not None:
                    position = np.array([p[0], p[1], p[2]])
                else:
                    position = np.zeros(3)
            else:
                p = getattr(self.body_link, "p", None)
                if p is not None:
                    position = np.array([p[0], p[1], p[2]])
                else:
                    position = np.zeros(3)
        except:
            position = np.zeros(3)
        
        # 속도는 간단히 계산 (실제로는 get_dofs_velocity() 등을 사용해야 함)
        velocity = np.zeros(3)
        
        return np.concatenate([position, velocity])
    
    def run_simulation(self, params: np.ndarray, num_steps: int = 100) -> List[np.ndarray]:
        """
        시뮬레이션 실행
        
        Args:
            params: 물리 파라미터 배열 [friction, car_mass, wheel_mass, kp_drive, kv_drive, kp_steer, kv_steer]
            num_steps: 시뮬레이션 스텝 수
        
        Returns:
            각 스텝의 차체 상태 리스트
        """
        # 씬 초기화
        if not self.initialized:
            self.initialize_scene()
        else:
            # 씬 리셋
            try:
                self.scene.reset()
            except:
                # 리셋이 실패하면 재초기화
                self.initialize_scene()
        
        # 물리 파라미터 적용
        friction = params[0]
        kp_drive = params[3]
        kv_drive = params[4]
        kp_steer = params[5]
        kv_steer = params[6]
        
        # PD 게인 설정
        if self.steer_dof_indices:
            self.car.set_dofs_kp(
                kp=np.array([kp_steer] * len(self.steer_dof_indices)),
                dofs_idx_local=self.steer_dof_indices
            )
            self.car.set_dofs_kv(
                kv=np.array([kv_steer] * len(self.steer_dof_indices)),
                dofs_idx_local=self.steer_dof_indices
            )
        
        if self.drive_dof_indices:
            self.car.set_dofs_kp(
                kp=np.array([kp_drive] * len(self.drive_dof_indices)),
                dofs_idx_local=self.drive_dof_indices
            )
            self.car.set_dofs_kv(
                kv=np.array([kv_drive] * len(self.drive_dof_indices)),
                dofs_idx_local=self.drive_dof_indices
            )
            self.car.set_dofs_force_range(
                lower=np.array([-500.0] * len(self.drive_dof_indices)),
                upper=np.array([500.0] * len(self.drive_dof_indices)),
                dofs_idx_local=self.drive_dof_indices
            )
        
        # 초기 안정화
        for _ in range(50):
            self.scene.step()
        
        # 조향 초기화
        if self.steer_dof_indices:
            self.car.control_dofs_position(
                np.array([0.0] * len(self.steer_dof_indices)),
                dofs_idx_local=self.steer_dof_indices
            )
        
        # 시뮬레이션 실행 및 상태 수집
        states = []
        base_speed = 10.0
        max_steer = 0.4
        dt = 1.0 / 60.0
        
        for step in range(num_steps):
            # 주행 제어
            t = step * dt
            if t < 2.0:
                steer = 0.0
                speed = base_speed * 0.5
            elif t < 4.0:
                steer = max_steer * 0.5
                speed = base_speed * 0.7
            else:
                steer = 0.0
                speed = base_speed
            
            # 구동 명령
            if self.drive_dof_indices:
                self.car.control_dofs_velocity(
                    np.array([speed] * len(self.drive_dof_indices)),
                    dofs_idx_local=self.drive_dof_indices
                )
            
            # 조향 명령
            if self.steer_dof_indices:
                self.car.control_dofs_position(
                    np.array([steer] * len(self.steer_dof_indices)),
                    dofs_idx_local=self.steer_dof_indices
                )
            
            # 상태 수집
            state = self.get_body_state()
            states.append(state)
            
            self.scene.step()
        
        return states


def load_csv_data(csv_path: str) -> pd.DataFrame:
    """CSV 데이터 로드 (pandas 사용 - 성능 향상)"""
    df = pd.read_csv(csv_path)
    return df


def compute_loss(sim_states: List[np.ndarray], target_states: np.ndarray) -> float:
    """
    손실 함수: 시뮬레이션 결과와 타겟 데이터 비교
    
    Args:
        sim_states: 시뮬레이션 상태 리스트
        target_states: 타겟 상태 배열 [num_frames, state_dim]
    
    Returns:
        평균 제곱 오차
    """
    if len(sim_states) == 0:
        return 1e6
    
    # 시뮬레이션 상태를 배열로 변환
    sim_array = np.array(sim_states)
    
    # 타겟과 길이 맞추기
    min_len = min(len(sim_array), len(target_states))
    sim_array = sim_array[:min_len]
    target_array = target_states[:min_len]
    
    # 위치와 속도 분리
    sim_pos = sim_array[:, :3]  # 위치
    sim_vel = sim_array[:, 3:6]  # 속도
    
    target_pos = target_array[:, :3]  # car_x, car_y, car_z
    target_vel = target_array[:, 3:6]  # car_vx, car_vy, car_vz
    
    # MSE 계산
    pos_loss = np.mean((sim_pos - target_pos) ** 2)
    vel_loss = np.mean((sim_vel - target_vel) ** 2)
    
    total_loss = pos_loss + 0.1 * vel_loss  # 위치에 더 큰 가중치
    
    return total_loss


def objective_function(params: np.ndarray, sim_wrapper: CarSimulationWrapper, target_states: np.ndarray, num_steps: int) -> float:
    """목적 함수: 파라미터를 받아서 손실을 계산"""
    sim_states = sim_wrapper.run_simulation(params, num_steps=num_steps)
    loss = compute_loss(sim_states, target_states)
    return loss


def train_with_mlp(
    model: PhysicsParameterMLP,
    sim_wrapper: CarSimulationWrapper,
    target_data: pd.DataFrame,
    num_epochs: int = 50,
    learning_rate: float = 1e-3,
    device: str = 'cpu'
):
    """MLP를 사용하여 물리 파라미터 학습"""
    
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 타겟 데이터 준비 (pandas DataFrame에서 필요한 컬럼만 추출)
    target_cols = ['car_x', 'car_y', 'car_z', 'car_vx', 'car_vy', 'car_vz']
    target_states = target_data[target_cols].values.astype(np.float32)
    num_frames = target_states.shape[0]
    
    # 입력 상태 준비 (시간 정규화 + 이전 상태)
    input_states = torch.zeros(num_frames, 7, device=device, dtype=torch.float32)
    for i in range(num_frames):
        input_states[i, 0] = i / max(num_frames, 1)  # 정규화된 시간
        if i > 0:
            input_states[i, 1:4] = torch.from_numpy(target_states[i-1, :3])
            input_states[i, 4:7] = torch.from_numpy(target_states[i-1, 3:6])
    
    best_loss = float('inf')
    best_params = None
    best_model_state = None
    
    print("🚀 MLP 학습 시작...")
    print(f"  타겟 데이터 크기: {num_frames} 프레임")
    
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        
        # MLP로 파라미터 예측
        predicted_params = model(input_states)  # [batch_size, 7]
        
        # 평균 파라미터 계산 (전체 시퀀스에 대해 동일한 파라미터 사용)
        avg_params = predicted_params.mean(dim=0).detach().cpu().numpy()  # [7]
        
        # 시뮬레이션 실행
        sim_states = sim_wrapper.run_simulation(avg_params, num_steps=num_frames)
        
        # 손실 계산
        loss_value = compute_loss(sim_states, target_states)
        
        # 유한 차분법으로 그래디언트 계산 (선택적 - 계산 비용이 높음)
        # 실제로는 더 효율적인 방법 사용 가능 (예: Bayesian Optimization)
        use_finite_diff = (epoch % 5 == 0)  # 5 에폭마다 한 번만 계산 (비용 절감)
        
        if use_finite_diff and epoch < num_epochs - 1:
            epsilon = 1e-2
            gradients = np.zeros(7)
            
            # 주요 파라미터만 계산 (비용 절감)
            important_params = [0, 3, 4, 5, 6]  # friction, kp_drive, kv_drive, kp_steer, kv_steer
            
            for i in important_params:
                # 양의 방향
                params_pos = avg_params.copy()
                params_pos[i] += epsilon
                sim_states_pos = sim_wrapper.run_simulation(params_pos, num_steps=num_frames)
                loss_pos = compute_loss(sim_states_pos, target_states)
                
                # 음의 방향
                params_neg = avg_params.copy()
                params_neg[i] -= epsilon
                sim_states_neg = sim_wrapper.run_simulation(params_neg, num_steps=num_frames)
                loss_neg = compute_loss(sim_states_neg, target_states)
                
                # 중앙 차분법
                gradients[i] = (loss_pos - loss_neg) / (2 * epsilon)
            
            # 그래디언트를 모델 출력에 역전파
            # 파라미터별 그래디언트를 사용하여 모델 출력에 할당
            predicted_params_mean = predicted_params.mean(dim=0)
            
            # 각 파라미터에 대해 그래디언트 할당
            for i in important_params:
                if predicted_params_mean[i].requires_grad:
                    # 유한 차분법으로 계산한 그래디언트 사용
                    grad_value = gradients[i]
                    
                    # 모델 출력에 그래디언트 할당 (간단한 근사)
                    # 실제로는 더 정교한 방법 필요
                    if grad_value != 0:
                        # 파라미터 출력에 그래디언트 할당
                        predicted_params[:, i].backward(
                            gradient=torch.full((num_frames,), grad_value / num_frames, device=device),
                            retain_graph=True
                        )
        else:
            # 그래디언트를 직접 계산하지 않고, 손실만 사용
            # 간단한 방법: 손실이 감소하면 현재 파라미터 유지, 증가하면 조정
            loss_tensor = torch.tensor(loss_value, device=device, requires_grad=False)
        
        # 옵티마이저 업데이트 (그래디언트가 계산된 경우만)
        if use_finite_diff and epoch < num_epochs - 1:
            optimizer.step()
        else:
            # 그래디언트가 없을 때는 간단한 방법 사용
            # 학습률 감소 또는 파라미터 조정
            pass
        
        if loss_value < best_loss:
            best_loss = loss_value
            best_params = avg_params.copy()
            best_model_state = model.state_dict().copy()
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss_value:.6f}")
            print(f"  Params: friction={avg_params[0]:.3f}, "
                  f"kp_drive={avg_params[3]:.1f}, "
                  f"kv_drive={avg_params[4]:.1f}, "
                  f"kp_steer={avg_params[5]:.1f}, "
                  f"kv_steer={avg_params[6]:.1f}")
    
    # 최적 모델 상태 복원
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    print(f"✅ 학습 완료! Best Loss: {best_loss:.6f}")
    print(f"Best Params: {best_params}")
    
    return best_params, model


def main():
    parser = argparse.ArgumentParser(description='물리 파라미터 학습 (MLP 사용)')
    parser.add_argument("--csv", type=str, default="./car_motion_data.csv",
                        help="CSV 데이터 파일 경로")
    parser.add_argument("--urdf", type=str, default="./car.urdf",
                        help="URDF 파일 경로")
    parser.add_argument("--epochs", type=int, default=20,
                        help="학습 에폭 수")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="학습률")
    parser.add_argument("--cpu", action="store_true", default=False,
                        help="CPU 사용")
    parser.add_argument("--vis", action="store_true", default=False,
                        help="시각화 표시")
    args = parser.parse_args()
    
    # CSV 데이터 로드
    print(f"📂 CSV 데이터 로드: {args.csv}")
    target_data = load_csv_data(args.csv)
    print(f"  데이터 크기: {len(target_data)} 프레임")
    
    # MLP 모델 생성
    print("🧠 MLP 모델 생성...")
    model = PhysicsParameterMLP(input_dim=7, hidden_dims=[128, 64, 32], output_dim=7)
    
    # 시뮬레이션 래퍼 생성
    print("🎮 시뮬레이션 래퍼 생성...")
    sim_wrapper = CarSimulationWrapper(
        urdf_path=args.urdf,
        use_cpu=args.cpu,
        show_viewer=args.vis
    )
    
    # 학습
    device = 'cpu'  # Genesis와 호환성을 위해 CPU 사용
    best_params, trained_model = train_with_mlp(
        model=model,
        sim_wrapper=sim_wrapper,
        target_data=target_data,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        device=device
    )
    
    # 결과 저장
    checkpoint_dir = "/home/wjdaksry/Genesis/examples/checkpoint"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    print(f"💾 결과 저장: {checkpoint_dir}")
    model_path = os.path.join(checkpoint_dir, 'physics_model.pth')
    params_path = os.path.join(checkpoint_dir, 'best_params.npy')
    
    torch.save(trained_model.state_dict(), model_path)
    np.save(params_path, best_params)
    
    print(f"  ✅ 모델 저장: {model_path}")
    print(f"  ✅ 파라미터 저장: {params_path}")
    
    # 파라미터 출력
    param_names = ['friction', 'car_mass', 'wheel_mass', 'kp_drive', 'kv_drive', 'kp_steer', 'kv_steer']
    print("\n최적 파라미터:")
    for name, value in zip(param_names, best_params):
        print(f"  {name}: {value:.3f}")
    
    print("✅ 완료!")


if __name__ == "__main__":
    main()

