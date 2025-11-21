import genesis as gs
import os
import argparse
import numpy as np
import math



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=True)
    parser.add_argument("-c", "--cpu", action="store_true", default=False)
    parser.add_argument("-p", "--parameter", type=str, default=None,
                        help="학습된 파라미터 파일 경로 (예: ./checkpoint/best_params.npy)")
    args = parser.parse_args()

    ########################## Init ##########################
    try:
        backend = gs.cpu if args.cpu else gs.gpu
        gs.init(backend=backend, logging_level="info")
        print(f"✅ Genesis initialized with backend: {backend}")
    except Exception as e:
        print(f"⚠️ GPU backend failed: {e}\n→ Switching to CPU")
        gs.init(backend=gs.cpu, logging_level="info")

    ########################## Scene ##########################
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            substeps=10,
            gravity=(0, 0, -9.81),
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(3, 2, 1.5),
            camera_lookat=(0, 0, 0.5),
            camera_up=(0, 0, 1),
        ),
        show_viewer=args.vis,
    )

    ########################## Materials ##########################
    # 파라미터 파일에서 friction을 미리 로드할 수 있으면 로드, 아니면 기본값 사용
    default_friction = 1.0
    if args.parameter and os.path.exists(args.parameter):
        try:
            best_params = np.load(args.parameter)
            if len(best_params) >= 7:
                default_friction = float(best_params[0])
        except:
            pass  # 기본값 사용
    
    rigid_mat = gs.materials.Rigid()
    ground_mat = gs.materials.Rigid(friction=default_friction)

    ########################## Add Ground ##########################
    ## ground 를 가장 prior로
    ground = scene.add_entity(
        morph=gs.morphs.Plane(),
        material=ground_mat
    )
    

    ########################## Load URDF ##########################
    # 🔹 기본 형상만 사용하는 URDF (mesh 없음)
    car_path = "/home/wjdaksry/Genesis/examples/car.urdf"

    if not os.path.exists(car_path):
        raise FileNotFoundError(f"❌ URDF not found: {car_path}")

    # 초기 위치: 바퀴가 지면에 닿도록 설정
    # 차체 중심을 0.8으로 설정하면:
    # - 차체 하단: 0.8 - 0.25 = 0.55
    # - 서스펜션: 0.55 - 0.25 = 0.3
    # - 조향: 0.3 - 0.35 = -0.05
    # - 바퀴 하단: -0.05 - 0.35 = -0.4 (여전히 지면 아래)
    # 기본 형상만 사용하므로 scale 불필요
    car = scene.add_entity(
        morph=gs.morphs.URDF(
            file="./car.urdf", 
            pos=(0, 0, 0.8),
            merge_fixed_links=False  # fixed 조인트 병합 방지 (에러 방지)
        ),
        material=rigid_mat
    )
    

    ########################## Build Scene ##########################
    scene.build()
    print("✅ Scene built successfully")


    ########################### drive #################################
        ############# drive (robust) ##############
    
    # 1) 조인트 이름 자동 탐색 함수 정의
    def joint_names_with(substr):
        found = []
        for j in getattr(car, "joints", []):
            if substr in getattr(j, "name", ""):
                found.append(j.name)
        return found

    # 2) 모든 조인트 출력 (디버깅)
    all_joints = [j.name for j in getattr(car, "joints", [])]
    print(f"[INFO] All joints: {all_joints}")

    # 3) 조인트 그룹화 및 DOF 인덱스 가져오기
    steer_joint_names = joint_names_with("steer") or ["steer_fl", "steer_fr"]
    rotate_joint_names = joint_names_with("rotate")
    rear_joint_names = joint_names_with("rear")
    drive_joint_names = rotate_joint_names + rear_joint_names
    if not drive_joint_names:
        drive_joint_names = ["rear_rl", "rear_rr", "wheel_fl_rotate", "wheel_fr_rotate"]

    print(f"[INFO] steer_joint_names: {steer_joint_names}")
    print(f"[INFO] drive_joint_names: {drive_joint_names}")

    # 조인트 이름으로 DOF 인덱스 가져오기
    def get_dof_indices(joint_names):
        dof_indices = []
        for name in joint_names:
            try:
                joint = car.get_joint(name)
                if joint and hasattr(joint, 'dofs_idx_local') and len(joint.dofs_idx_local) > 0:
                    dof_indices.append(joint.dofs_idx_local[0])
                    print(f"[INFO] Joint '{name}' -> DOF index: {joint.dofs_idx_local[0]}")
                else:
                    print(f"[WARN] Joint '{name}' not found or has no DOF")
            except Exception as e:
                print(f"[WARN] Failed to get joint '{name}': {e}")
        return dof_indices

    steer_dof_indices = get_dof_indices(steer_joint_names)
    drive_dof_indices = get_dof_indices(drive_joint_names)
    
    if not steer_dof_indices:
        print("[ERROR] No steer joints found!")
    if not drive_dof_indices:
        print("[ERROR] No drive joints found!")
    
    print(f"[INFO] steer_dof_indices: {steer_dof_indices}")
    print(f"[INFO] drive_dof_indices: {drive_dof_indices}")
    
    # 파라미터 로드 및 적용
    # 기본값 설정
    default_friction = 1.0
    default_kp_steer = 1000.0
    default_kv_steer = 100.0
    default_kp_drive = 500.0
    default_kv_drive = 50.0
    default_car_mass = None  # URDF에서 가져온 값 사용
    default_wheel_mass = None  # URDF에서 가져온 값 사용
    
    # 학습된 파라미터 로드 (지정된 경우)
    if args.parameter:
        params_path = args.parameter
        if os.path.exists(params_path):
            try:
                best_params = np.load(params_path)
                if len(best_params) >= 7:
                    friction, car_mass, wheel_mass, kp_drive, kv_drive, kp_steer, kv_steer = best_params[:7]
                    
                    print(f"📂 학습된 파라미터 로드: {params_path}")
                    print(f"  friction: {friction:.3f}")
                    print(f"  car_mass: {car_mass:.1f}")
                    print(f"  wheel_mass: {wheel_mass:.1f}")
                    print(f"  kp_drive: {kp_drive:.1f}, kv_drive: {kv_drive:.1f}")
                    print(f"  kp_steer: {kp_steer:.1f}, kv_steer: {kv_steer:.1f}")
                    
                    # Ground friction 적용 (scene.build() 이후에는 material 속성 변경이 제한될 수 있음)
                    # 이미 초기화 시점에 friction이 설정되었으므로 추가 조치 불필요
                    if abs(ground.material.friction - friction) > 1e-3:
                        try:
                            ground.material.friction = float(friction)
                            print(f"  ✅ Ground friction 업데이트: {friction:.3f}")
                        except Exception as e:
                            print(f"  ⚠️ Ground friction 업데이트 실패 (이미 scene.build() 이후): {e}")
                            print(f"     다음 실행 시 --parameter 옵션으로 friction이 적용됩니다.")
                    
                    # 차체 및 바퀴 질량 적용 (선택사항)
                    if car_mass > 0 and car.get_link("car_body"):
                        try:
                            car.get_link("car_body").set_mass(float(car_mass))
                            print(f"  ✅ 차체 질량 적용: {car_mass:.1f} kg")
                        except Exception as e:
                            print(f"  ⚠️ 차체 질량 적용 실패: {e}")
                    
                    for wheel_name in ["wheel_fl", "wheel_fr", "wheel_rl", "wheel_rr"]:
                        if wheel_mass > 0 and car.get_link(wheel_name):
                            try:
                                car.get_link(wheel_name).set_mass(float(wheel_mass))
                            except Exception as e:
                                pass  # 조용히 실패
                    
                    # 기본값을 학습된 값으로 교체
                    default_friction = friction
                    default_kp_steer = kp_steer
                    default_kv_steer = kv_steer
                    default_kp_drive = kp_drive
                    default_kv_drive = kv_drive
                else:
                    print(f"⚠️ 파라미터 파일 형식 오류: 7개 값이 필요하지만 {len(best_params)}개만 있음")
                    print("  기본값 사용")
            except Exception as e:
                print(f"⚠️ 파라미터 파일 로드 실패: {e}")
                print("  기본값 사용")
        else:
            print(f"⚠️ 파라미터 파일을 찾을 수 없음: {params_path}")
            print("  기본값 사용")
    else:
        print("ℹ️ 파라미터 파일이 지정되지 않음. 기본값 사용")
    
    # PD 게인 설정 (학습된 값 또는 기본값 사용)
    if steer_dof_indices:
        car.set_dofs_kp(
            kp=np.array([default_kp_steer] * len(steer_dof_indices)),  # 위치 게인
            dofs_idx_local=steer_dof_indices
        )
        car.set_dofs_kv(
            kv=np.array([default_kv_steer] * len(steer_dof_indices)),  # 속도 게인
            dofs_idx_local=steer_dof_indices
        )
    
    if drive_dof_indices:
        car.set_dofs_kp(
            kp=np.array([default_kp_drive] * len(drive_dof_indices)),  # 위치 게인 (낮게, 속도 제어용)
            dofs_idx_local=drive_dof_indices
        )
        car.set_dofs_kv(
            kv=np.array([default_kv_drive] * len(drive_dof_indices)),  # 속도 게인
            dofs_idx_local=drive_dof_indices
        )
        # 힘 범위 설정 (안전을 위해)
        car.set_dofs_force_range(
            lower=np.array([-5000.0] * len(drive_dof_indices)),  # 더 큰 힘 범위 허용
            upper=np.array([5000.0] * len(drive_dof_indices)),
            dofs_idx_local=drive_dof_indices
        )

    # 5) 초기 안정화 (서스펜션이 fixed이므로 짧은 안정화만 필요)
    print("⏳ 초기 안정화 중...")
    
    # 서스펜션이 fixed이므로 초기화 불필요
    # 짧은 안정화 시간만 사용
    for i in range(100):
        scene.step()
        if i % 50 == 0:
            print(f"  안정화 진행: {i}/100")
    
    print("✅ 초기 안정화 완료")

    # 3) body 링크 얻기 (이름 우선, 실패 시 첫 링크)
    def get_body_link(entity):
        prefer = ("base","car_body","base_link","chassis","body")
        links = getattr(entity, "links", []) or []
        for name in prefer:
            for L in links:
                if getattr(L, "name", "") == name:
                    return L
        return links[0] if links else None

    body_link = get_body_link(car)

    # 4) pose 읽기 시도 (버전별로 다른 함수 지원)
    def get_body_pose():
        """Return (position, rotation_matrix) if available."""
        if body_link is None:
            return None

        # ✅ Genesis 공식 API: get_world_transform()만 사용 (pose 호출 금지!)
        fn = getattr(body_link, "get_world_transform", None)
        if callable(fn):
            try:
                pose = fn()
                p = getattr(pose, "p", None)
                r = getattr(pose, "r", None)
                if p is not None and r is not None:
                    return p, r
            except Exception as e:
                print(f"[WARN] Failed to get transform: {e}")
                return None

        # fallback (예전 버전일 경우)
        try:
            p = getattr(body_link, "p", None)
            r = getattr(body_link, "r", None)
            if p is not None and r is not None:
                return p, r
        except:
            pass

        return None

    # 6) 조향 초기화 (위치 0으로)
    if steer_dof_indices:
        car.control_dofs_position(
            np.array([0.0] * len(steer_dof_indices)),
            dofs_idx_local=steer_dof_indices
        )

    print("🚗 Drive simulation started...")

    # === 모드: 시간 기반 주행 (간단한 테스트) ===
    base_speed = 100.0           # 속도 조정 (rad/s, 바퀴 회전 속도)
    max_steer  = 0.4            # 최대 조향 각도 (rad)
    dt = 1.0 / 60.0             # 시간 스텝 (대략 60fps)

    steps = 3000

    for step in range(steps):
        # 시간 기반 주행 제어
        t = step * dt
        if t < 2.0:
            steer = 0.0
            speed = base_speed * 0.5
        elif t < 4.0:
            steer = max_steer * 0.5  # 약한 좌회전
            speed = base_speed * 0.7
        elif t < 6.0:
            steer = 0.0
            speed = base_speed
        else:
            steer = -max_steer * 0.3  # 약한 우회전
            speed = base_speed * 0.8

        # 구동 명령: 모든 바퀴에 회전 속도 적용
        if drive_dof_indices:
            car.control_dofs_velocity(
                np.array([speed] * len(drive_dof_indices)),
                dofs_idx_local=drive_dof_indices
            )
        
        # 조향 명령: 앞바퀴만
        if steer_dof_indices:
            car.control_dofs_position(
                np.array([steer] * len(steer_dof_indices)),
                dofs_idx_local=steer_dof_indices
            )
        
        if step % 200 == 0:  # 200스텝마다 로그
            print(f"[DEBUG] Step {step}: speed={speed:.2f} rad/s, steer={steer:.3f} rad")
            if drive_dof_indices:
                print(f"  - Drive DOFs: {drive_dof_indices}")
            if steer_dof_indices:
                print(f"  - Steer DOFs: {steer_dof_indices}")

        scene.step()

    print("✅ Drive simulation finished.")

    ########################## Simulation Loop ##########################
    


if __name__ == "__main__":
    main()

