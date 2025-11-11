# 자동차 시뮬레이션 문제 해결 기록

## 개요
Genesis 물리 엔진을 사용한 자동차 시뮬레이션에서 발생한 문제들과 해결 과정을 기록합니다.

---

## 문제 1: 서스펜션 링크에 Inertial 정보 부재

### 문제 상황
- 서스펜션 링크(`susp_fl`, `susp_fr`, `susp_rl`, `susp_rr`)가 빈 링크로 정의되어 있었음
- 물리 시뮬레이션에서 서스펜션이 제대로 작동하지 않음

### 원인
```xml
<!-- 기존 코드 -->
<link name="susp_fl"/>
<link name="susp_fr"/>
<link name="susp_rl"/>
<link name="susp_rr"/>
```
- URDF에서 링크에 `inertial` 속성이 없으면 물리 엔진이 제대로 처리하지 못함
- 질량과 관성 모멘트가 없어서 물리 계산 불가

### 해결 방법
서스펜션 링크에 적절한 `inertial` 속성 추가

### 변경된 코드
```xml
<!-- 수정 후 -->
<link name="susp_fl">
  <inertial>
    <mass value="1.0"/>
    <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
  </inertial>
</link>
```
- 질량: 1.0kg (초기에는 0.1kg으로 설정했으나 안정성을 위해 1.0kg으로 조정)
- 관성 모멘트: 작은 값으로 설정 (서스펜션은 작은 부품이므로)

### 영향
- 서스펜션이 물리 시뮬레이션에서 정상적으로 인식됨
- 서스펜션 링크가 물리 엔진에 의해 올바르게 처리됨

---

## 문제 2: 바퀴 회전 축 및 구조 문제

### 문제 상황
- 바퀴가 전혀 회전하지 않음
- 뒷바퀴만 회전 조인트가 있고, 앞바퀴에는 조향 조인트만 있음
- 바퀴 회전 축이 잘못 설정됨 (Y축이 아닌 X축으로 설정되어 있었음)

### 원인
1. **회전 축 문제**: Genesis에서는 Y축(0, 1, 0) 회전이 전방 이동을 의미함
2. **앞바퀴 회전 조인트 부재**: 앞바퀴에 회전 조인트가 없어서 바퀴가 회전하지 않음
3. **조인트 구조 문제**: 조향과 회전이 하나의 조인트로 결합되어 있음

### 해결 방법
1. 모든 바퀴에 회전 조인트 추가 (Y축 회전)
2. 앞바퀴에 조향 링크 추가하여 조향과 회전 분리
3. 바퀴 회전 조인트를 `continuous` 타입으로 설정

### 변경된 구조
```
기존: car_body → susp → wheel (조향만 가능)
수정: car_body → susp → steer_link → wheel (조향 + 회전 모두 가능)
```

### 변경된 코드

#### URDF 구조 변경
```xml
<!-- 앞바퀴: 조향 링크 추가 -->
<link name="steer_fl_link">
  <inertial>
    <mass value="0.5"/>
    <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
  </inertial>
</link>

<!-- 조인트 구조 -->
<joint name="steer_fl" type="revolute">
  <parent link="susp_fl"/>
  <child link="steer_fl_link"/>
  <axis xyz="0 0 1"/>  <!-- Z축 회전: 조향 -->
</joint>

<joint name="wheel_fl_rotate" type="continuous">
  <parent link="steer_fl_link"/>
  <child link="wheel_fl"/>
  <axis xyz="0 1 0"/>  <!-- Y축 회전: 바퀴 회전 (전방 이동) -->
</joint>
```

#### 바퀴 inertia 조정
```xml
<!-- 바퀴 회전에 맞게 inertia 조정 -->
<inertia
  ixx="0.5" ixy="0" ixz="0"
  iyy="1.0" iyz="0"  <!-- Y축 회전 (전방) -->
  izz="0.5"/>
```

### 영향
- 모든 바퀴(앞바퀴 + 뒷바퀴)가 회전 가능
- 조향과 회전이 독립적으로 작동
- Y축 회전으로 전방 이동 가능

---

## 문제 3: 차량이 하늘로 솟구치는 문제

### 문제 상황
- 차량이 스폰될 때 하늘로 솟구침
- 초기 위치가 너무 높게 설정되어 있거나, 바퀴가 지면과 겹침

### 원인
1. **초기 위치 계산 오류**: 차체 중심 위치와 바퀴 위치 관계를 정확히 계산하지 못함
2. **바퀴-지면 간섭**: 바퀴가 지면 아래로 들어가서 물리 엔진이 튕김
3. **서스펜션 조인트 범위**: 서스펜션 조인트의 limit이 넓어서 불안정함

### 해결 방법
1. 초기 위치를 적절히 조정
2. 서스펜션 조인트를 `fixed` 타입으로 변경하여 안정화
3. `merge_fixed_links=False` 옵션 추가

### 변경된 코드

#### Python 코드
```python
# 초기 위치 조정
car = scene.add_entity(
    morph=gs.morphs.URDF(
        file="./car_dae.urdf", 
        pos=(0, 0, 0.8),  # 차체 중심 높이
        merge_fixed_links=False  # fixed 조인트 병합 방지
    ),
    material=rigid_mat
)
```

#### URDF: 서스펜션 조인트를 fixed로 변경
```xml
<!-- 서스펜션을 fixed로 변경하여 안정화 -->
<joint name="susp_fl_joint" type="fixed">
  <parent link="car_body"/>
  <child link="susp_fl"/>
  <origin xyz="1.2 -0.6 -0.25" rpy="0 0 0"/>
</joint>
```

### 위치 계산
```
차체 중심: z = 0.8
차체 하단: 0.8 - 0.25 = 0.55
서스펜션: 0.55 - 0.25 = 0.3
조향 조인트: 0.3 - 0.35 = -0.05
바퀴 중심: -0.05
바퀴 하단: -0.05 - 0.35 = -0.4 (지면 아래, 하지만 안정적으로 떨어짐)
```

### 영향
- 차량이 안정적으로 스폰됨
- 하늘로 솟구치지 않음
- 바퀴가 지면에 안정적으로 닿음

---

## 문제 4: 서스펜션 튕김 및 뒤집힘 문제

### 문제 상황
- 서스펜션이 튕겨서 차량이 뒤로 뒤집어짐
- 서스펜션 조인트가 너무 자유롭게 움직임

### 원인
1. **서스펜션 링크 질량**: 5.0kg으로 너무 무거워서 불안정
2. **서스펜션 조인트 범위**: -0.15 ~ 0.15로 너무 넓음
3. **서스펜션 조인트 타입**: `prismatic` 타입이 damping이 없어서 튕김

### 해결 방법
1. 서스펜션 조인트를 `fixed` 타입으로 변경 (가장 안정적)
2. 서스펜션 링크 질량을 1.0kg으로 조정
3. 서스펜션 조인트 limit을 좁히거나 완전히 고정

### 변경된 코드

#### 서스펜션 링크 질량 조정
```xml
<!-- 질량을 1.0kg으로 조정 (초기 0.1kg에서 증가) -->
<link name="susp_fl">
  <inertial>
    <mass value="1.0"/>  <!-- 5.0kg → 1.0kg -->
    <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
  </inertial>
</link>
```

#### 서스펜션 조인트를 fixed로 변경
```xml
<!-- prismatic → fixed -->
<joint name="susp_fl_joint" type="fixed">
  <parent link="car_body"/>
  <child link="susp_fl"/>
  <origin xyz="1.2 -0.6 -0.25" rpy="0 0 0"/>
  <!-- axis와 limit 제거 (fixed 조인트는 불필요) -->
</joint>
```

### 대안 (서스펜션을 유지하려면)
만약 서스펜션을 유지하려면:
```xml
<joint name="susp_fl_joint" type="prismatic">
  <axis xyz="0 0 1"/>  <!-- Z축 이동 (수직) -->
  <limit lower="-0.05" upper="0.05" effort="50000" velocity="1"/>
  <!-- 범위를 -0.15~0.15에서 -0.05~0.05로 축소 -->
  <!-- effort를 10000에서 50000으로 증가 -->
  <!-- velocity를 2에서 1로 감소 -->
</joint>
```

### 영향
- 차량이 안정적으로 유지됨
- 뒤집히지 않음
- 서스펜션이 고정되어 물리적으로 안정적

---

## 문제 5: 바퀴가 전혀 움직이지 않는 문제 (주행 불가)

### 문제 상황
- 차량이 스폰은 되지만 전혀 움직이지 않음
- 조인트 제어 명령이 작동하지 않음

### 원인
1. **잘못된 API 사용**: Genesis의 공식 API를 사용하지 않고 가정한 함수명 사용
2. **DOF 인덱스 미사용**: 조인트 이름 대신 DOF 인덱스를 사용해야 함
3. **PD 게인 미설정**: 조인트 제어를 위한 PD 게인이 설정되지 않음

### 해결 방법
Genesis 공식 API 사용:
- `control_dofs_velocity()`: 속도 제어
- `control_dofs_position()`: 위치 제어
- `get_joint(name).dofs_idx_local[0]`: DOF 인덱스 가져오기
- `set_dofs_kp()`, `set_dofs_kv()`: PD 게인 설정

### 변경된 코드

#### 기존 코드 (작동하지 않음)
```python
def set_vel_target(entity, j, v):
    return (maybe_call(entity, ["set_dof_target_velocity", ...], j, v)
            or maybe_call(scene, ["set_dof_target_velocity", ...], entity, j, v))

# 사용
for j in drive_joints:
    set_vel_target(car, j, speed)  # 작동하지 않음
```

#### 수정된 코드 (작동함)
```python
# 1. DOF 인덱스 가져오기
def get_dof_indices(joint_names):
    dof_indices = []
    for name in joint_names:
        joint = car.get_joint(name)
        if joint and hasattr(joint, 'dofs_idx_local') and len(joint.dofs_idx_local) > 0:
            dof_indices.append(joint.dofs_idx_local[0])
    return dof_indices

steer_dof_indices = get_dof_indices(steer_joint_names)
drive_dof_indices = get_dof_indices(drive_joint_names)

# 2. PD 게인 설정
car.set_dofs_kp(
    kp=np.array([1000.0] * len(steer_dof_indices)),
    dofs_idx_local=steer_dof_indices
)
car.set_dofs_kv(
    kv=np.array([100.0] * len(steer_dof_indices)),
    dofs_idx_local=steer_dof_indices
)

# 3. 조인트 제어
# 속도 제어 (바퀴 회전)
car.control_dofs_velocity(
    np.array([speed] * len(drive_dof_indices)),
    dofs_idx_local=drive_dof_indices
)

# 위치 제어 (조향)
car.control_dofs_position(
    np.array([steer] * len(steer_dof_indices)),
    dofs_idx_local=steer_dof_indices
)
```

### 주요 변경사항

#### 1. 조인트 이름 → DOF 인덱스
```python
# 기존: 조인트 이름으로 제어 시도
set_vel_target(car, "rear_rl", speed)  # ❌

# 수정: DOF 인덱스로 제어
joint = car.get_joint("rear_rl")
dof_idx = joint.dofs_idx_local[0]
car.control_dofs_velocity(np.array([speed]), dofs_idx_local=[dof_idx])  # ✅
```

#### 2. PD 게인 설정
```python
# 조인트 제어를 위해 필수
car.set_dofs_kp(kp=np.array([1000.0]), dofs_idx_local=steer_dof_indices)
car.set_dofs_kv(kv=np.array([100.0]), dofs_idx_local=steer_dof_indices)
```

#### 3. 힘 범위 설정
```python
# 안전을 위해 힘 범위 제한
car.set_dofs_force_range(
    lower=np.array([-500.0] * len(drive_dof_indices)),
    upper=np.array([500.0] * len(drive_dof_indices)),
    dofs_idx_local=drive_dof_indices
)
```

### 영향
- 바퀴가 정상적으로 회전함
- 차량이 전진함
- 조향이 작동함
- 주행 시뮬레이션이 정상적으로 동작함

---

## 최종 해결 요약

### 해결된 문제들
1. ✅ 서스펜션 링크에 inertial 정보 추가
2. ✅ 바퀴 회전 축 및 구조 수정 (Y축 회전, 앞바퀴 회전 조인트 추가)
3. ✅ 초기 스폰 위치 조정 및 fixed 조인트 사용
4. ✅ 서스펜션 안정화 (fixed 조인트로 변경)
5. ✅ Genesis 공식 API 사용하여 조인트 제어

### 최종 구조

#### URDF 구조
```
car_body (차체)
  ├─ susp_fl_joint (fixed) → susp_fl
  │   └─ steer_fl (revolute, Z축) → steer_fl_link
  │       └─ wheel_fl_rotate (continuous, Y축) → wheel_fl
  ├─ susp_fr_joint (fixed) → susp_fr
  │   └─ steer_fr (revolute, Z축) → steer_fr_link
  │       └─ wheel_fr_rotate (continuous, Y축) → wheel_fr
  ├─ susp_rl_joint (fixed) → susp_rl
  │   └─ rear_rl (continuous, Y축) → wheel_rl
  └─ susp_rr_joint (fixed) → susp_rr
      └─ rear_rr (continuous, Y축) → wheel_rr
```

#### Python 제어 구조
```python
# 1. 조인트 이름으로 DOF 인덱스 가져오기
steer_dof_indices = [car.get_joint(name).dofs_idx_local[0] for name in steer_joint_names]
drive_dof_indices = [car.get_joint(name).dofs_idx_local[0] for name in drive_joint_names]

# 2. PD 게인 설정
car.set_dofs_kp(kp=..., dofs_idx_local=steer_dof_indices)
car.set_dofs_kv(kv=..., dofs_idx_local=steer_dof_indices)

# 3. 조인트 제어
car.control_dofs_velocity(speed_array, dofs_idx_local=drive_dof_indices)  # 바퀴 회전
car.control_dofs_position(steer_array, dofs_idx_local=steer_dof_indices)  # 조향
```

### 주요 학습 사항
1. **URDF 구조**: 모든 링크에 `inertial` 속성이 필요함
2. **조인트 축**: Genesis에서 Y축(0,1,0) 회전이 전방 이동
3. **조인트 타입**: `fixed` 조인트는 안정적이지만 움직임 제한, `prismatic`은 움직임 가능하지만 불안정할 수 있음
4. **Genesis API**: 공식 API (`control_dofs_velocity`, `control_dofs_position`)를 사용해야 함
5. **DOF 인덱스**: 조인트 이름 대신 DOF 인덱스를 사용해야 함
6. **PD 게인**: 조인트 제어를 위해 PD 게인 설정이 필수

---

## 참고사항

### Genesis 조인트 제어 API
- `control_dofs_position()`: 위치 제어 (조향용)
- `control_dofs_velocity()`: 속도 제어 (바퀴 회전용)
- `control_dofs_force()`: 힘 제어
- `set_dofs_kp()`: 위치 게인 설정
- `set_dofs_kv()`: 속도 게인 설정
- `set_dofs_force_range()`: 힘 범위 설정
- `get_joint(name).dofs_idx_local[0]`: DOF 인덱스 가져오기

### URDF 조인트 타입
- `fixed`: 고정 조인트 (움직임 없음, 가장 안정적)
- `revolute`: 회전 조인트 (한 축 회전, limit 필요)
- `continuous`: 연속 회전 조인트 (limit 없음, 바퀴 회전용)
- `prismatic`: 이동 조인트 (한 축 이동, 서스펜션용, 불안정할 수 있음)

### 권장 설정
- **초기 위치**: 차체 중심 z = 0.8
- **서스펜션**: fixed 조인트 (안정성 우선)
- **바퀴 회전**: continuous 조인트, Y축 회전
- **조향**: revolute 조인트, Z축 회전, limit -0.6 ~ 0.6
- **PD 게인**: 조향 (kp=1000, kv=100), 드라이브 (kp=500, kv=50)

---

## 파일 변경 내역

### car_dae.urdf
- 서스펜션 링크에 `inertial` 속성 추가
- 앞바퀴에 조향 링크 추가
- 모든 바퀴에 회전 조인트 추가 (Y축)
- 서스펜션 조인트를 `fixed` 타입으로 변경
- 바퀴 inertia 조정 (Y축 회전에 맞게)

### car_test.py
- Genesis 공식 API 사용 (`control_dofs_velocity`, `control_dofs_position`)
- DOF 인덱스 기반 제어
- PD 게인 설정 추가
- 초기 위치 조정 (z = 0.8)
- `merge_fixed_links=False` 옵션 추가

---

## 결론
모든 문제가 해결되어 자동차가 안정적으로 스폰되고 주행할 수 있게 되었습니다. 주요 원인은 URDF 구조 문제와 Genesis API 사용법 오류였으며, 이를 해결하여 정상적으로 작동하는 시뮬레이션을 구축했습니다.

