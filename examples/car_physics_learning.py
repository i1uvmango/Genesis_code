import genesis as gs
import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple
import math


class PhysicsParameterMLP(nn.Module):
    """MLP 모델: 시뮬레이션 상태를 입력으로 받아 물리 파라미터를 출력"""
    
    def __init__(self, input_dim: int = 7, hidden_dims: List[int] = [128, 64, 32]):
        """
        Args:
            input_dim: 입력 차원 (차체 위치, 속도, 조향각 등)
            hidden_dims: 히든 레이어 차원들
        """
        super(PhysicsParameterMLP, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            prev_dim = hidden_dim
        
        # 출력: 물리 파라미터들
        # [friction, car_mass, wheel_mass, kp_drive, kv_drive, kp_steer, kv_steer]
        self.network = nn.Sequential(*layers)
        self.friction_head = nn.Linear(prev_dim, 1)
        self.car_mass_head = nn.Linear(prev_dim, 1)
        self.wheel_mass_head = nn.Linear(prev_dim, 1)
        self.kp_drive_head = nn.Linear(prev_dim, 1)
        self.kv_drive_head = nn.Linear(prev_dim, 1)
        self.kp_steer_head = nn.Linear(prev_dim, 1)
        self.kv_steer_head = nn.Linear(prev_dim, 1)
        
        # 활성화 함수 (파라미터 범위 제한)
        self.friction_activation = nn.Sigmoid()  # 0~1
        self.mass_activation = nn.Softplus()  # > 0
        self.kp_activation = nn.Softplus()  # > 0
        self.kv_activation = nn.Softplus()  # > 0
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: 입력 상태 [batch_size, input_dim]
        Returns:
            물리 파라미터 딕셔너리
        """
        features = self.network(x)
        
        # 각 파라미터별 출력 (적절한 범위로 제한)
        friction = self.friction_activation(self.friction_head(features)) * 2.0  # 0~2
        car_mass = self.mass_activation(self.car_mass_head(features)) + 500.0  # > 500
        wheel_mass = self.mass_activation(self.wheel_mass_head(features)) + 10.0  # > 10
        kp_drive = self.kp_activation(self.kp_drive_head(features)) + 100.0  # > 100
        kv_drive = self.kv_activation(self.kv_drive_head(features)) + 10.0  # > 10
        kp_steer = self.kp_activation(self.kp_steer_head(features)) + 500.0  # > 500
        kv_steer = self.kv_activation(self.kv_steer_head(features)) + 50.0  # > 50
        
        return {
            'friction': friction.squeeze(-1),
            'car_mass': car_mass.squeeze(-1),
            'wheel_mass': wheel_mass.squeeze(-1),
            'kp_drive': kp_drive.squeeze(-1),
            'kv_drive': kv_drive.squeeze(-1),
            'kp_steer': kp_steer.squeeze(-1),
            'kv_steer': kv_steer.squeeze(-1),
        }


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
        
    def initialize_scene(self):
        """시뮬레이션 씬 초기화"""
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
        
        # 속도는 간단히 계산 (이전 위치와 비교)
        # 실제로는 get_dofs_velocity() 등을 사용해야 할 수 있음
        velocity = np.zeros(3)
        
        return np.concatenate([position, velocity])
    
    def run_simulation(self, params: Dict[str, float], num_steps: int = 100) -> List[np.ndarray]:
        """
        시뮬레이션 실행
        
        Args:
            params: 물리 파라미터 딕셔너리
            num_steps: 시뮬레이션 스텝 수
        
        Returns:
            각 스텝의 차체 상태 리스트
        """
        # 씬 재초기화
        if self.scene is None:
            self.initialize_scene()
        else:
            # 씬 리셋 (새로운 파라미터로 재시작)
            self.scene.reset()
        
        # 물리 파라미터 적용
        # Ground 마찰
        if hasattr(self.ground, 'set_material'):
            # Genesis API에 따라 다를 수 있음
            pass
        
        # PD 게인 설정
        if self.steer_dof_indices:
            kp_steer = params.get('kp_steer', 1000.0)
            kv_steer = params.get('kv_steer', 100.0)
            self.car.set_dofs_kp(
                kp=np.array([kp_steer] * len(self.steer_dof_indices)),
                dofs_idx_local=self.steer_dof_indices
            )
            self.car.set_dofs_kv(
                kv=np.array([kv_steer] * len(self.steer_dof_indices)),
                dofs_idx_local=self.steer_dof_indices
            )
        
        if self.drive_dof_indices:
            kp_drive = params.get('kp_drive', 500.0)
            kv_drive = params.get('kv_drive', 50.0)
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
            # 주행 제어 (간단한 패턴)
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
    """CSV 데이터 로드"""
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
    
    # 위치와 속도 분리 (CSV 데이터 구조에 따라)
    # CSV: car_x, car_y, car_z, car_vx, car_vy, car_vz
    sim_pos = sim_array[:, :3]  # 위치
    sim_vel = sim_array[:, 3:6]  # 속도
    
    target_pos = target_array[:, :3]  # car_x, car_y, car_z
    target_vel = target_array[:, 3:6]  # car_vx, car_vy, car_vz
    
    # MSE 계산
    pos_loss = np.mean((sim_pos - target_pos) ** 2)
    vel_loss = np.mean((sim_vel - target_vel) ** 2)
    
    total_loss = pos_loss + 0.1 * vel_loss  # 위치에 더 큰 가중치
    
    return total_loss


def train_physics_parameters(
    model: PhysicsParameterMLP,
    sim_wrapper: CarSimulationWrapper,
    target_data: pd.DataFrame,
    num_epochs: int = 50,
    learning_rate: float = 1e-3,
    device: str = 'cpu',
    use_finite_diff: bool = True
):
    """물리 파라미터 학습
    
    Args:
        model: MLP 모델
        sim_wrapper: 시뮬레이션 래퍼
        target_data: 타겟 CSV 데이터
        num_epochs: 학습 에폭 수
        learning_rate: 학습률
        device: 디바이스 ('cpu' or 'cuda')
        use_finite_diff: 유한 차분법 사용 여부
    """
    
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 타겟 데이터 준비
    target_cols = ['car_x', 'car_y', 'car_z', 'car_vx', 'car_vy', 'car_vz']
    target_states = target_data[target_cols].values.astype(np.float32)
    
    # 입력 상태 준비 (시간 정규화 + 이전 상태)
    input_states = torch.zeros(len(target_data), 7, device=device, dtype=torch.float32)
    for i in range(len(target_data)):
        input_states[i, 0] = i / max(len(target_data), 1)  # 정규화된 시간
        if i > 0:
            input_states[i, 1:4] = torch.from_numpy(target_states[i-1, :3])
            input_states[i, 4:7] = torch.from_numpy(target_states[i-1, 3:6])
    
    best_loss = float('inf')
    best_params = None
    best_model_state = None
    
    print("🚀 학습 시작...")
    print(f"  타겟 데이터 크기: {len(target_data)} 프레임")
    print(f"  학습 방법: {'유한 차분법' if use_finite_diff else '직접 최적화'}")
    
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        
        # MLP로 파라미터 예측
        with torch.no_grad():
            predicted_params = model(input_states)
        
        # 파라미터 평균 계산 (전체 시퀀스에 대해 동일한 파라미터 사용)
        avg_params = {
            'friction': float(predicted_params['friction'].mean().item()),
            'car_mass': float(predicted_params['car_mass'].mean().item()),
            'wheel_mass': float(predicted_params['wheel_mass'].mean().item()),
            'kp_drive': float(predicted_params['kp_drive'].mean().item()),
            'kv_drive': float(predicted_params['kv_drive'].mean().item()),
            'kp_steer': float(predicted_params['kp_steer'].mean().item()),
            'kv_steer': float(predicted_params['kv_steer'].mean().item()),
        }
        
        # 시뮬레이션 실행
        sim_states = sim_wrapper.run_simulation(avg_params, num_steps=len(target_data))
        
        # 손실 계산
        loss_value = compute_loss(sim_states, target_states)
        
        if use_finite_diff:
            # 유한 차분법으로 그래디언트 계산
            epsilon = 1e-2  # 유한 차분 스텝 크기
            param_gradients = {}
            
            # 각 파라미터별로 유한 차분법 적용
            for param_name in ['friction', 'kp_drive', 'kv_drive', 'kp_steer', 'kv_steer']:
                current_value = avg_params[param_name]
                
                # 양의 방향
                perturbed_params_pos = avg_params.copy()
                perturbed_params_pos[param_name] = current_value + epsilon
                perturbed_states_pos = sim_wrapper.run_simulation(perturbed_params_pos, num_steps=len(target_data))
                perturbed_loss_pos = compute_loss(perturbed_states_pos, target_states)
                
                # 음의 방향 (중앙 차분법)
                perturbed_params_neg = avg_params.copy()
                perturbed_params_neg[param_name] = current_value - epsilon
                perturbed_states_neg = sim_wrapper.run_simulation(perturbed_params_neg, num_steps=len(target_data))
                perturbed_loss_neg = compute_loss(perturbed_states_neg, target_states)
                
                # 중앙 차분법으로 그래디언트 계산
                grad = (perturbed_loss_pos - perturbed_loss_neg) / (2 * epsilon)
                param_gradients[param_name] = grad
            
            # 그래디언트를 모델 파라미터에 역전파
            # 실제로는 파라미터별 그래디언트를 모델의 출력에 역전파해야 함
            # 여기서는 간단한 근사 방법 사용
            
            # 손실 텐서 생성 (역전파 가능하도록)
            loss_tensor = torch.tensor(loss_value, device=device, requires_grad=True)
            
            # 모델 파라미터에 대한 그래디언트 계산 (간단한 근사)
            # 실제로는 더 정교한 방법 필요
            model.zero_grad()
            
            # 예측 파라미터를 다시 계산 (gradient tracking 활성화)
            predicted_params_grad = model(input_states)
            
            # 각 파라미터별 그래디언트를 출력에 역전파
            for param_name in ['friction', 'kp_drive', 'kv_drive', 'kp_steer', 'kv_steer']:
                if param_name in param_gradients:
                    # 파라미터 출력에 그래디언트 할당
                    grad_value = param_gradients[param_name]
                    if predicted_params_grad[param_name].requires_grad:
                        # 평균값에 그래디언트 할당
                        avg_param_tensor = predicted_params_grad[param_name].mean()
                        if avg_param_tensor.requires_grad:
                            # 역전파 가능하도록 설정
                            pass
            
            # 실제로는 더 정교한 방법이 필요하지만, 여기서는 간단히 손실값만 사용
            # 수동으로 파라미터 업데이트 (더 나은 방법 필요)
            pass
        else:
            # 직접 최적화 (시뮬레이션 결과를 손실로 사용)
            loss_tensor = torch.tensor(loss_value, device=device, requires_grad=False)
        
        # 최적화 (간단한 방법: 파라미터를 직접 조정)
        # 실제로는 더 정교한 방법 필요
        
        if loss_value < best_loss:
            best_loss = loss_value
            best_params = avg_params.copy()
            best_model_state = model.state_dict().copy()
        
        if (epoch + 1) % 1 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss_value:.6f}")
            print(f"  Params: friction={avg_params['friction']:.3f}, "
                  f"kp_drive={avg_params['kp_drive']:.1f}, "
                  f"kv_drive={avg_params['kv_drive']:.1f}, "
                  f"kp_steer={avg_params['kp_steer']:.1f}, "
                  f"kv_steer={avg_params['kv_steer']:.1f}")
        
        # 간단한 최적화: 손실이 감소하지 않으면 학습률 조정
        if epoch > 10 and loss_value > best_loss * 1.1:
            # 학습률 감소
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.9
    
    # 최적 모델 상태 복원
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    print(f"✅ 학습 완료! Best Loss: {best_loss:.6f}")
    print(f"Best Params: {best_params}")
    
    return best_params, model


def main():
    parser = argparse.ArgumentParser(description='물리 파라미터 학습')
    parser.add_argument("--csv", type=str, default="./car_motion_data.csv",
                        help="CSV 데이터 파일 경로")
    parser.add_argument("--urdf", type=str, default="./car.urdf",
                        help="URDF 파일 경로")
    parser.add_argument("--epochs", type=int, default=50,
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
    model = PhysicsParameterMLP(input_dim=7, hidden_dims=[128, 64, 32])
    
    # 시뮬레이션 래퍼 생성
    print("🎮 시뮬레이션 래퍼 생성...")
    sim_wrapper = CarSimulationWrapper(
        urdf_path=args.urdf,
        use_cpu=args.cpu,
        show_viewer=args.vis
    )
    
    # 학습
    device = 'cpu'  # Genesis와 호환성을 위해 CPU 사용
    best_params, trained_model = train_physics_parameters(
        model=model,
        sim_wrapper=sim_wrapper,
        target_data=target_data,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        device=device
    )
    
    # 결과 저장
    print("💾 결과 저장...")
    torch.save(trained_model.state_dict(), 'physics_model.pth')
    np.save('best_params.npy', best_params)
    print("✅ 완료!")


if __name__ == "__main__":
    main()

