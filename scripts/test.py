import time

import mujoco
import mujoco.viewer

class PDController:
    def __init__(self, kp, kd, setpoint):
        self.kp = kp
        self.kd = kd
        self.setpoint = setpoint
        self.prev_error = 0

    def compute(self, measured_value):
        error = self.setpoint - measured_value
        derivative = error - self.prev_error
        output = (self.kp * error) + (self.kd * derivative)
        self.prev_error = error
        return output



m = mujoco.MjModel.from_xml_path('mujoco_menagerie-main/skydio_x2/x2.xml')
d = mujoco.MjData(m)
d.ctrl[:4] = 4  # Set the thruster values to 0.5

kp = 0.5  # Proportional gain
kd = 0.1  #
setpoint = 0 # vertical speed should be 0

pd_controller = PDController(kp, kd, setpoint)

with mujoco.viewer.launch_passive(m, d) as viewer:
  
  # Close the viewer automatically after 30 wall-seconds.
  start = time.time()
  while viewer.is_running() and time.time() - start < 30:
    step_start = time.time()
    
    ### Control loop to stabelize drone height.
    measured_speed = d.qvel[2]
    #print (measured_speed)
    control_signal = pd_controller.compute(measured_speed)
    print(control_signal)
    d.ctrl[:4] = d.ctrl[:4] + control_signal
    # mj_step can be replaced with code that also evaluates
    # a policy and applies a control signal before stepping the physics.
    mujoco.mj_step(m, d)

    # Example modification of a viewer option: toggle contact points every two seconds.
    with viewer.lock():
      viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(d.time % 2)

    # Pick up changes to the physics state, apply perturbations, update options from GUI.
    viewer.sync()

    # Rudimentary time keeping, will drift relative to wall clock.
    time_until_next_step = m.opt.timestep - (time.time() - step_start)
    if time_until_next_step > 0:
      time.sleep(time_until_next_step)