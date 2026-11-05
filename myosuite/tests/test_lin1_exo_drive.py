# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""The optional Lin-I hip drive must behave like the DC motor it claims to be.

``myoLegWalkExo-v0(exo_motor=True)`` hangs a native MuJoCo ``<dcmotor>`` on each exo hip
flexion hinge, so the torque the device delivers is produced by the electromechanical
relation ``tau = N*K*(V - K*N*omega) / R`` rather than by a clip on an ideal torque source.
That is only worth having if the element actually reproduces the datasheet, so these tests
pin down the four things a bench measurement would check: the torque-speed line, the peak
torque cap, the reflected rotor inertia, and the winding current implied by the torque.

The drive is a RobStride RS02, and the parameters in ``lin1_exo_params.json`` are fitted from
its ``使用说明书 261009``, so two things are verified here: that the model is faithful to the
numbers it is given, and that those numbers still describe the published device.  The default
env stays exactly as passive as it was: the drive is opt-in and never enters the policy action.
"""

from __future__ import annotations

import dataclasses

import mujoco
import numpy as np
import pytest

import myosuite  # noqa: F401  (registers the CPU envs)
from myosuite import make_env
from myosuite.envs.myo.tasks.basic.leg.lin1_exo import (
    EXO_PARAMS_PATH,
    ExoMotorConfig,
    load_exo_params,
)
from myosuite.envs.myo.tasks.basic.leg.walk_exo import ExoAssistConfig

pytestmark = pytest.mark.tier1

EXO_ENV = "myoLegWalkExo-v0"
SIDES = ("r", "l")
FLEXION = tuple(f"exo_hip_flexion_{s}" for s in SIDES)


def _sensor(env, name: str, width: int = 1) -> np.ndarray:
    adr = int(env.model.sensor_adr[env.model.sensor(name).id])
    return env.data.sensordata[adr : adr + width]


def _hinge_dof(env, joint: str) -> int:
    return int(env.model.jnt_dofadr[env.model.joint(joint).id])


def _drive_torque(env, side: str = "r") -> float:
    """The joint-side torque the drive delivered on the last step, N*m.

    ``exo_motor_torque_{side}`` is an ACTUATORFRC readout, which MuJoCo reports at the
    *rotor* shaft, so it has to be multiplied by the gear ratio to become a joint torque.
    """
    return float(_sensor(env, f"exo_motor_torque_{side}")[0]) * motor.gear_ratio


def _drive_at(env, voltage: float, joint_speed: float, side: str = "r") -> float:
    """Delivered *joint-side* torque, N*m, for one hinge held at ``joint_speed``."""
    ctrl = env._exo_ctrl_adr[int(SIDES.index(side))]
    env.data.qvel[:] = 0.0
    env.data.qvel[_hinge_dof(env, f"exo_hip_flexion_{side}")] = joint_speed
    env.data.ctrl[ctrl] = voltage
    mujoco.mj_step(env.model, env.data)
    return _drive_torque(env, side)


motor = ExoMotorConfig.from_params(load_exo_params(EXO_PARAMS_PATH))


@pytest.fixture(scope="module")
def passive_env():
    return make_env(EXO_ENV).unwrapped


@pytest.fixture(scope="module")
def powered_env():
    env = make_env(EXO_ENV, exo_motor=True).unwrapped
    env.reset(seed=0)
    return env


# -- the drive is opt-in ------------------------------------------------------------------
def test_the_default_exo_env_stays_passive(passive_env, powered_env) -> None:
    assert passive_env.model.nu == powered_env.model.nu - len(SIDES)
    assert passive_env.model.nsensor == powered_env.model.nsensor - 4 * len(SIDES)
    assert passive_env.exo_motor_max_torques().size == powered_env.exo_dof_positions().size
    assert np.all(passive_env.exo_motor_max_torques() == 0.0)


def test_the_drive_never_enters_the_policy_action(powered_env, passive_env) -> None:
    assert powered_env.action_space.shape == passive_env.action_space.shape
    # the two voltage slots are appended after the muscles, so a muscle-only action cannot
    # reach them even by accident
    assert list(powered_env._exo_ctrl_adr) == [
        powered_env.model.nu - 2,
        powered_env.model.nu - 1,
    ]
    powered_env.reset(seed=0)
    powered_env.set_exo_voltage([7.0, -7.0])
    powered_env.step(np.ones(powered_env.action_space.shape, dtype=np.float32))
    assert np.all(powered_env.data.ctrl[powered_env._exo_ctrl_adr] == [7.0, -7.0])


def test_the_drive_leaves_the_observation_the_baseline_policy_expects(
    powered_env, passive_env
) -> None:
    """Fitting the drive may not widen what the policy sees: that transfer is the design.

    ``muscle_length/velocity/force`` are read off ``data.actuator_*``, which is indexed by
    actuator rather than by muscle, so the two motor slots are appended to them unless the
    env strips them back out -- six numbers a published 403-dim checkpoint has no weights
    for, and the failure is silent because gymnasium accepts whatever shape comes back.
    """
    powered_obs, _ = powered_env.reset(seed=0)
    passive_obs, _ = passive_env.reset(seed=0)
    assert powered_obs.shape == passive_obs.shape == passive_env.observation_space.shape
    assert np.array_equal(powered_obs, passive_obs), (
        "the drive changed the observation, so no published policy can drive this env")


def test_reset_leaves_the_hip_drive_unpowered(powered_env) -> None:
    powered_env.set_exo_voltage([motor.bus_voltage, motor.bus_voltage])
    powered_env.reset(seed=0)
    assert np.all(powered_env.data.ctrl[powered_env._exo_ctrl_adr] == 0.0)
    # an unpowered hinge still resists: with zero applied voltage the back-EMF term turns
    # any motion into a braking torque, so "unpowered" is not "free"
    powered_env.data.qvel[_hinge_dof(powered_env, FLEXION[0])] = 2.0
    mujoco.mj_forward(powered_env.model, powered_env.data)
    mujoco.mj_step(powered_env.model, powered_env.data)
    assert _drive_torque(powered_env, "r") == pytest.approx(
        motor.envelope(0.0, 2.0), abs=1e-9
    )
    assert _drive_torque(powered_env, "r") < 0.0


def test_voltage_is_clamped_to_the_bus(powered_env) -> None:
    powered_env.set_exo_voltage([4.0 * motor.bus_voltage, -4.0 * motor.bus_voltage])
    cr = powered_env.model.actuator_ctrlrange[powered_env._exo_ctrl_adr]
    ctrl = powered_env.data.ctrl[powered_env._exo_ctrl_adr]
    assert ctrl[0] == pytest.approx(cr[0, 1])
    assert ctrl[1] == pytest.approx(cr[1, 0])
    assert np.all(cr[:, 1] == motor.bus_voltage)
    assert np.all(cr[:, 0] == -motor.bus_voltage)


# -- the electromechanical relation -------------------------------------------------------
@pytest.mark.parametrize(
    "voltage, joint_speed",
    [
        (motor.bus_voltage, 0.0),
        (motor.bus_voltage * 0.5, 0.0),
        (12.0, 0.0),
        (motor.bus_voltage, 1.0),
        (motor.bus_voltage, 3.0),
        (20.0, 2.0),
        (-motor.bus_voltage, 0.0),
        (-20.0, 1.5),
    ],
)
def test_delivered_torque_sits_on_the_torque_speed_line(
    powered_env, voltage: float, joint_speed: float
) -> None:
    powered_env.reset(seed=0)
    got = _drive_at(powered_env, voltage, joint_speed)
    assert got == pytest.approx(motor.envelope(voltage, joint_speed), abs=1e-9)


def test_the_peak_torque_cap_binds_before_the_stall_torque(powered_env) -> None:
    # the un-clamped electrical line would give more than the gearbox is rated for
    unclamped = (
        motor.gear_ratio * motor.torque_constant * motor.bus_voltage
        / motor.effective_resistance
    )
    assert unclamped > motor.peak_joint_torque
    powered_env.reset(seed=0)
    assert _drive_at(powered_env, motor.bus_voltage, 0.0) == pytest.approx(
        motor.peak_joint_torque, abs=1e-9
    )
    # ...and the cap is applied at the motor shaft, which is where it physically lives
    assert float(_sensor(powered_env, "exo_motor_torque_r")[0]) == pytest.approx(
        motor.rotor_torque_limit, abs=1e-9
    )


def test_back_emf_rolls_off_the_torque_with_speed(powered_env) -> None:
    powered_env.reset(seed=0)
    slow = _drive_at(powered_env, 20.0, 0.0)
    fast = _drive_at(powered_env, 20.0, 2.0)
    assert slow - fast == pytest.approx(2.0 * abs(motor.torque_slope), rel=1e-9)
    assert slow > fast > 0.0


def test_a_request_over_the_bus_returns_the_rail_voltage_and_a_shortfall() -> None:
    # Fast enough that the electrical line, not the gearbox, is what binds: the no-load
    # speed is 42.9 rad/s, so at 30 the envelope has already fallen under the 17 N*m cap.
    speed = 30.0
    ceiling = motor.envelope(motor.bus_voltage, speed)
    assert ceiling < motor.peak_joint_torque
    volts = motor.voltage_for_torque(motor.peak_joint_torque, speed)
    assert volts == pytest.approx(motor.bus_voltage)
    # what the drive delivers is back on the line, and the gap is the unserved request
    assert motor.envelope(volts, speed) == pytest.approx(ceiling)
    assert ceiling < motor.peak_joint_torque
    # the part it *can* serve still round-trips exactly
    assert motor.envelope(
        motor.voltage_for_torque(ceiling, speed), speed
    ) == pytest.approx(ceiling, abs=1e-9)


@pytest.mark.parametrize("joint_speed", [0.0, 1.0, 3.0, 8.0])
def test_the_torque_request_round_trips_through_the_simulated_drive(
    powered_env, joint_speed: float
) -> None:
    """The controller asks for N*m; the inversion has to land that request in the sim.

    Checked against the ``<dcmotor>`` element itself rather than against
    :meth:`ExoMotorConfig.envelope`, so an inversion that only agrees with its own formula
    still fails.
    """
    powered_env.reset(seed=0)
    ceiling = motor.envelope(motor.bus_voltage, joint_speed)
    for wanted in (-ceiling, -0.5 * ceiling, 0.0, 0.3 * ceiling, ceiling):
        volts = motor.voltage_for_torque(wanted, joint_speed)
        assert abs(volts) <= motor.bus_voltage
        assert _drive_at(powered_env, volts, joint_speed) == pytest.approx(wanted, abs=1e-9)


def test_winding_current_matches_the_rotor_torque(powered_env) -> None:
    powered_env.reset(seed=0)
    powered_env.set_exo_voltage([motor.bus_voltage, 0.0])
    powered_env.data.qvel[:] = 0.0
    mujoco.mj_step(powered_env.model, powered_env.data)
    state = powered_env.exo_motor_state()
    assert state["current"][0] == pytest.approx(
        motor.rotor_torque_limit / motor.torque_constant, rel=1e-9
    )
    assert state["current"][1] == pytest.approx(0.0, abs=1e-12)


# -- the mechanical side of the drive -----------------------------------------------------
def test_rotor_inertia_is_reflected_onto_the_hinge_as_gear_squared(powered_env) -> None:
    bare = make_env(EXO_ENV, exo_motor=True, exo_motor_overrides={"rotor_inertia": 0.0})
    bare.unwrapped.reset(seed=0)
    powered_env.reset(seed=0)
    dof = _hinge_dof(powered_env, FLEXION[0])

    def diagonal(env) -> float:
        dense = np.zeros((env.model.nv, env.model.nv))
        mujoco.mj_forward(env.model, env.data)
        mujoco.mj_fullM(env.model, env.data, dense)
        return float(dense[dof, dof])

    assert diagonal(powered_env) - diagonal(bare.unwrapped) == pytest.approx(
        motor.gear_ratio ** 2 * motor.rotor_inertia, abs=1e-9
    )
    bare.close()


# -- the configuration itself -------------------------------------------------------------
def _rpm(w: float) -> float:
    return w * 30.0 / np.pi


def test_every_drive_number_declares_where_it_came_from() -> None:
    block = load_exo_params(EXO_PARAMS_PATH)["actuator"]
    assert block.get("uncalibrated") is False
    assert motor.calibrated is True, "a datasheet block must not still claim to be a placeholder"
    assert block["datasheet"]["document"].strip(), "the numbers need a citable source"
    assert block["datasheet"]["retrieved"].strip(), "... and a date, because they get revised"
    estimated = set(block.get("estimated_fields", []))
    assert estimated <= set(block), "estimated_fields must name fields the block actually has"
    assert "rotor_inertia_kg_m2" in estimated, (
        "the one drive number the RS02 document does not give has to stay flagged")
    assert block["calibration_note"].strip(), "the file must say what still needs measuring"


def test_the_fitted_curve_is_the_rs02_curve_and_not_a_plausible_one() -> None:
    """Pin the calibration itself: the fitted line must reproduce the published 48 V envelope.

    ``RS02 使用说明书 261009`` gives a 17 N.m current-limited cap, 410 rpm no-load and a
    T-N line running from the cap to zero, so a block that still passes the physics tests
    above can nevertheless have drifted off the device.
    """
    v = motor.bus_voltage
    assert _rpm(motor.no_load_joint_speed) == pytest.approx(410.0, rel=0.02)
    assert motor.envelope(v, 0.0) == pytest.approx(17.0)
    assert motor.envelope(v, 100 * 2 * np.pi / 60) == pytest.approx(17.0)
    assert motor.envelope(v, 300 * 2 * np.pi / 60) == pytest.approx(10.1, abs=0.8)
    # the driver's 23 Apk is what makes 17 N.m the ceiling: the unclamped line is far above it
    unclamped = motor.gear_ratio * motor.torque_constant * v / motor.effective_resistance
    assert unclamped > 2.0 * motor.peak_joint_torque
    assert _rpm(unclamped / abs(motor.torque_slope)) == pytest.approx(410.0, rel=0.02)
    # and the model's current readout should agree with that cap
    assert motor.peak_current == pytest.approx(23.0 / np.sqrt(2.0), rel=0.12)


@pytest.mark.parametrize(
    "field, value, message",
    [
        ("gear_ratio", 0.0, "positive"),
        ("gear_efficiency", 0.0, r"\(0, 1\]"),
        ("gear_efficiency", 1.5, r"\(0, 1\]"),
        ("winding_resistance", -1.0, "positive"),
        ("peak_joint_torque", 0.0, "positive"),
        ("continuous_joint_torque", 500.0, "exceed"),
    ],
)
def test_incoherent_drive_numbers_are_rejected(field: str, value: float, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        dataclasses.replace(motor, **{field: value})


# -- the assist controller -----------------------------------------------------------------
# Push-off assist: negative extends the hip.  The amplitude is not the module's rating but the
# most the *cuff* can deliver: it transfers ~0.12 of the hinge torque to the hip, so this
# already asks the drive for its whole 17 N.m.
ASSIST_TABLE = load_exo_params(EXO_PARAMS_PATH)["assist"]["hip_coupling"]["hip_torque_transfer"]
ASSIST_TRANSFER = np.array([ASSIST_TABLE[s] for s in SIDES])
ASSIST = -motor.peak_joint_torque * float(np.abs(ASSIST_TRANSFER).min())


def _assisted(steps: int = 100, **kwargs):
    flags = {"exo_motor": True, "exo_assist_torque": ASSIST}
    flags.update(kwargs)
    env = make_env(EXO_ENV, **flags).unwrapped
    env.reset(seed=0)
    action = np.zeros(env.action_space.shape, dtype=np.float32)
    for _ in range(steps):
        env.step(action)
    return env, env.assist_summary()


def test_the_assist_controller_is_off_until_it_is_asked_for(powered_env) -> None:
    assert powered_env._exo_assist is None
    powered_env.step(np.zeros(powered_env.action_space.shape, dtype=np.float32))
    # without a controller nothing may touch the voltage slots
    assert np.all(powered_env.data.ctrl[powered_env._exo_ctrl_adr] == 0.0)
    assert powered_env.assist_summary()["phase"].size == 0


def test_the_assist_cannot_be_installed_without_the_drive_it_commands() -> None:
    with pytest.raises(ValueError, match="exo_motor"):
        make_env(EXO_ENV, exo_assist_torque=ASSIST)


def test_the_pulse_is_locked_to_the_gait_clock_of_each_side() -> None:
    env = make_env(EXO_ENV, exo_motor=True, exo_assist_torque=ASSIST).unwrapped
    env.reset(seed=0)
    cfg = env._exo_assist
    action = np.zeros(env.action_space.shape, dtype=np.float32)
    for _ in range(env.hip_period):
        env.step(action)
    log = env.assist_summary()
    phase, hip = log["phase"], log["commanded_hip_torque"]
    # measured extension peaks, advanced by the configured lead
    for k, side in enumerate(SIDES):
        center = (env._exo_assist_extension[k] - cfg.phase_lead) % 1.0
        peak = float(phase[int(np.argmax(np.abs(hip[:, k])))])
        # the plateau of a raised cosine is broad, so a step either side of the centre is fine
        assert abs((peak - center + 0.5) % 1.0 - 0.5) < 0.5 * cfg.width
        assert abs(hip[:, k]).max() == pytest.approx(abs(cfg.hip_flexion_torque), rel=0.02)
    # both legs get the same event half a cycle apart, not the same clock time
    right = float(phase[int(np.argmax(np.abs(hip[:, 0])))])
    left = float(phase[int(np.argmax(np.abs(hip[:, 1])))])
    assert abs((right - left + 0.5) % 1.0 - 0.5) == pytest.approx(0.5, abs=0.05)


def test_both_hips_are_assisted_in_the_same_direction() -> None:
    """The one thing the joint names cannot tell you: the two hinges turn opposite ways.

    The delivered hinge torque, referred back to the hip through the measured transfer, has
    to point where the controller asked - on both sides.  Read off the hinge columns alone
    the two sides look like they are fighting each other, and that is exactly the mistake
    this pins down.
    """
    env, log = _assisted(steps=100)
    hip, delivered = log["commanded_hip_torque"], log["delivered_hinge_torque"]
    at_hip = delivered * env._exo_assist_transfer
    live = np.abs(hip) > 0.5 * abs(ASSIST)  # the edges of the pulse ride up on the leg itself
    assert live.sum(axis=0).min() > 10, "the pulse barely came up: nothing was tested"
    assert np.all(np.sign(at_hip[live]) == np.sign(hip[live])), (
        "one leg is being resisted instead of assisted"
    )


def test_the_controller_uses_the_recorded_coupling_and_nothing_else() -> None:
    block = load_exo_params(EXO_PARAMS_PATH)["assist"]["hip_coupling"]
    transfer = np.array([block["hip_torque_transfer"][s] for s in SIDES])
    env = make_env(EXO_ENV, exo_motor=True, exo_assist_torque=ASSIST).unwrapped
    assert np.array_equal(env._exo_assist_transfer, transfer)
    assert transfer[0] * transfer[1] < 0, "the recorded transfer has to keep the opposite signs"
    assert np.all(np.abs(transfer) > 0.02), "a near-zero transfer would divide the command away"
    _, log = _assisted(steps=60)
    assert np.allclose(log["commanded_hinge_torque"],
                       log["commanded_hip_torque"] / transfer, rtol=1e-9)


def test_the_cuff_not_the_motor_sets_the_ceiling() -> None:
    """The assist is limited by the interface, and the code must not pretend otherwise.

    17 N.m at the hinge is about 2 N.m at the hip, a tenth of what the module can do, because
    the cuff is a point connection that carries force and no moment.  A config that asked for
    the module's rating would be asking for something the wearing arrangement cannot give.
    """
    ceiling = motor.peak_joint_torque * float(np.abs(ASSIST_TRANSFER).min())
    assert ceiling < motor.continuous_joint_torque / 2.0, (
        "the interface got stiffer; re-measure hip_torque_transfer before trusting this")
    assert abs(ASSIST) == pytest.approx(ceiling, rel=0.01)
    assert abs(ExoAssistConfig().hip_flexion_torque) <= ceiling + 1e-9, (
        "the shipped default asks for more than the cuff can deliver")


def test_a_zero_amplitude_control_actually_commands_zero_torque() -> None:
    """The comparison's control condition: drive fitted, controller running, nothing pushed.

    Holding a voltage of zero is *not* this: a shorted winding turns the hinge into a
    brake, which is why the zero-amplitude command is back-EMF compensation instead.
    """
    _, log = _assisted(steps=100, exo_assist_torque=0.0)
    assert np.all(log["commanded_hip_torque"] == 0.0)
    assert np.abs(log["delivered_hinge_torque"]).max() < 1.0, (
        "the idle drive is damping the swing, so a later gain cannot be blamed on the legs")


def test_an_over_request_is_reported_as_saturation_instead_of_silently_limited() -> None:
    # a request past the 17 N.m cap: the log has to show the gap, not just a clipped number
    _, log = _assisted(steps=40, exo_assist_torque=-4.0 * motor.peak_joint_torque,
                       exo_assist_phase_lead=0.0)
    assert log["saturated"].any()
    worst = np.abs(log["delivered_hinge_torque"]).max()
    assert worst <= motor.peak_joint_torque + 1e-6
    assert np.all(np.abs(log["voltage"]) <= motor.bus_voltage + 1e-9)
    peak = np.abs(log["commanded_hinge_torque"]).max()
    assert peak > motor.peak_joint_torque, "the request never left the envelope"
