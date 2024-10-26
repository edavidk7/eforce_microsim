class PID:
    def __init__(self, kp, ki, kd, max_output, min_output):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.max = max_output
        self.min = min_output
        self.i = 0.
        self.e_prev = 0.
        self.prev_process_variable = 0.

    def reset(self):
        self.i = 0.
        self.e_prev = 0.
        self.prev_process_variable = 0.

    def set_gains(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd

    def get_pid_gains(self, speed):
        """
        Adjust PID gains based on current speed or setpoint.
        This is the gain scheduling function.
        """
        self.set_gains(self.kp, self.ki, self.kd)

    def step(self, setpoint, process_variable, dt):
        """
        PID controller step with dynamic gain scheduling.

        Args:
          setpoint - desired value (speed)
          process_variable - current value (actual speed)
          dt - time step

        Returns:
            output - PID output
        """
        # Adjust PID gains based on current setpoint (or actual speed)
        self.get_pid_gains(setpoint)  # Optionally use process_variable instead of setpoint
        # Regulation step with anti-windup
        error = setpoint - process_variable 


        #elf.kp += 0.01
        #self.ki += 0.001
        #self.kd -= 0.0001


        #self.kd = self.kd * (process_variable-self.prev_process_variable) * dt * 100

        # Update integral (i) term with a limit to avoid windup
        integral_limit = 75  # Set a reasonable limit to avoid large integrals
        self.i += error * dt / 2
        self.i = max(min(self.i, integral_limit), -integral_limit)  # Apply the limit

        # PID control output
        output = self.kp * error + self.ki * self.i + self.kd * (error - self.e_prev) / dt

        # Save the previous error for derivative calculation
        self.e_prev = error

        # Clamp output to max/min limits
        output = max(self.min, min(output, self.max))

        self.prev_process_variable = process_variable

        return output
