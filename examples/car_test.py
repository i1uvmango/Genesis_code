import genesis as gs
import os
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=True)
    parser.add_argument("-c", "--cpu", action="store_true", default=False)
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
    rigid_mat = gs.materials.Rigid()

    ########################## Load URDF ##########################
    # 🔹 경로 수정 (새로 만든 안전한 URDF)
    car_path = "/home/wjdaksry/Genesis/examples/car_dae.urdf"

    if not os.path.exists(car_path):
        raise FileNotFoundError(f"❌ URDF not found: {car_path}")

    car = scene.add_entity(
        morph=gs.morphs.URDF(file="./car_dae.urdf"),

        material=rigid_mat,
    )

    ########################## Add Ground ##########################
    ground = scene.add_entity(
        morph=gs.morphs.Plane(),
        material=rigid_mat,
    )
    

    ########################## Build Scene ##########################
    scene.build()
    print("✅ Scene built successfully")

    ########################## Simulation Loop ##########################
    horizon = 1000 if "PYTEST_VERSION" not in os.environ else 5
    for i in range(horizon):
        scene.step()

if __name__ == "__main__":
    main()
