#include <Portenta_H7_PWM.h> 

// TODO create a motor object with the required pins or at the very least put all the pins in corresponding arrays.

// Define PWM PINS
#define MOTOR_X_PWM D6     // PWM0
#define MOTOR_Y_PWM D2     // PWM4 // won't allow me to use PWM3
#define MOTOR_Z_PWM D0     // PWM6

// Define motor channels pins(for position)

#define MotorXChA PC_15    // GPIO_1
#define MotorXChB PD_4     // GPIO_2
#define MotorYChA PD_5     // GPIO 3
#define MotorYChB PE_3     // GPIO 4
#define MotorZChA PG_3     // GPIO 5
#define MotorZChB PG_10    // GPIO 6

// MOTOR direction pins
#define Motor_X_dir1 PC_6  //UART0_TX //PWM1 pin   //PREV D4  // works, located at PWM5 pin
#define Motor_X_dir2 PC_7  //UART0_RX //PWM2 pin   //PREV A0
#define Motor_Y_dir1 D3    //PWM 3, PWM3 didn't work as output, donät know why.
#define Motor_Y_dir2 PK_1  //PWM 5 // D13 works, located at uart1 RX. VALUE OF D13 IS 13, NEED TO CHECK which pin is which on breakout.
#define Motor_Z_dir1 PJ_7  //PWM7 // GPIO_1
#define Motor_Z_dir2 PJ_10 //PWM8

// Fixed parameters
#define MAX_FREQ 10000.0f
#define DUTY_CYCLE_MAX 100.0f
#define DUTY_CYCLE_MIN 3.0f
#define MOVE_THRESHOLD 2
//#define SPEED_SCALE = 0.000000125f//1.25e-07 // 1/8000000
// PID parameters
#define K_P 5000.0f // 10_000f
#define K_I 0.0f // 0 previously
#define K_D 2000.0f // 1000 previously.

#define K_P_position 0.08f
#define K_D_position 0.08f


uint32_t motorDir1Pins[3] = { Motor_X_dir1, Motor_Y_dir1, Motor_Z_dir1 };
uint32_t motorDir2Pins[3] = { Motor_X_dir2, Motor_Y_dir2, Motor_Z_dir2 };
uint32_t motorPWMPins[3] = { MOTOR_X_PWM, MOTOR_Y_PWM, MOTOR_Z_PWM };

#define NUM_OF_PINS (sizeof(motorPWMPins) / sizeof(uint32_t))

// float dutyCycle[] = { 0.0f, 0.0f, 0.0f };  // Set to zero later
int motorMoveDirections[] = { 0, 0, 0 };  //+1 forward, -1 backward 0 stop

long motorPreviousPositions[] = { 32768, 32768, 32768 };  // int should be enough

const double SPEED_SCALE = 8000000.0f; // was 8000000.0f; with old algorithm //ol value was 25500000.0f; // convert from double to something in the same order as the typical ticks/microseconds of moving motor

unsigned long curTime[] = { 0, 0, 0 };
unsigned long prevTime[] = { 0, 0, 0 };
long dt = 1;
double ds = 0;  // speed error, delta speed


double previous_errors[3] = {0, 0, 0}; // store the previous errors for each motor
double integral_errors[3] = {0, 0, 0}; // store the integral of the errors for each motor

mbed::PwmOut* motor_pwm[] = { NULL, NULL, NULL };

int dist = 0;
int moveDist = 0; // distance to target position

// Error terms for PID

long motorTargetSpeedsPosition[3] = {0, 0, 0};
double integralError[3] = {0, 0, 0}; 

int getDist() {
  return dist;
}
long getDt() {
  return dt;
}


void initiateInterrupts() {

  for (uint8_t i = 0; i < 3; i++) {
    pinMode(motorDir1Pins[i], OUTPUT);  // something wrong with these actions?
    pinMode(motorDir2Pins[i], OUTPUT);
  }
  pinMode(MotorXChA, INPUT);  // CHA motor X, used to use INPUT_PULLUP
  pinMode(MotorXChB, INPUT);  // CHB motor X, used to use just INPUT.
  attachInterrupt(digitalPinToInterrupt(MotorXChA), X_CHA_trig, RISING);

  pinMode(MotorYChA, INPUT);  // CHA motor Y
  pinMode(MotorYChB, INPUT);  // CHB motor Y
  attachInterrupt(digitalPinToInterrupt(MotorYChA), Y_CHA_trig, RISING);

  pinMode(MotorZChA, INPUT);  // CHA motor Z
  pinMode(MotorZChB, INPUT);  // CHB motor Z
  attachInterrupt(digitalPinToInterrupt(MotorZChA), Z_CHA_trig, RISING);
}

void startAllPWM() {
  curTime[0] = micros();
  curTime[1] = micros();
  curTime[2] = micros();

  prevTime[0] = micros();
  prevTime[1] = micros();
  prevTime[2] = micros();

  for (uint8_t index = 0; index < NUM_OF_PINS; index++) {
    setPWM(motor_pwm[index], motorPWMPins[index], MAX_FREQ, dutyCycle[index]);
  }
}
void X_CHA_trig() {
  if (GPIOD->IDR & (1U << 4)) {
    motorPositions[0] += 1;
  } else{
    motorPositions[0] -= 1;
  }
}

void Y_CHA_trig() {
  if (GPIOE->IDR & (1U << 3)) { // CH_B
    motorPositions[1] += 1;
  } else{
    motorPositions[1] -= 1;
  }
}
void Z_CHA_trig() {
  if (GPIOG->IDR & (1U << 10))  {
    motorPositions[2] += 1;
  } else{
    motorPositions[2] -= 1;
  }
}

void moveForward(uint8_t motor_id) {
  // TODO check which is forward and which is backward
  if (motor_id == 0){
    digitalWrite(Motor_X_dir2, HIGH);
    digitalWrite(Motor_X_dir1, LOW); 
    return ;
  }
  if (motor_id == 1){
    digitalWrite(Motor_Y_dir2, HIGH);
    digitalWrite(Motor_Y_dir1, LOW); 
    return ;
  }
  if (motor_id == 2){
    digitalWrite(Motor_Z_dir2, HIGH);
    digitalWrite(Motor_Z_dir1, LOW); 
  }
}

void moveBackward(uint8_t motor_id) {
    // TODO replace digital write and read with direct registry manipulations.
    if (motor_id == 0){
    digitalWrite(Motor_X_dir2, LOW);
    digitalWrite(Motor_X_dir1, HIGH); 
    return ;
  }
  if (motor_id == 1){
    digitalWrite(Motor_Y_dir2, LOW);
    digitalWrite(Motor_Y_dir1, HIGH); 
    return ;
  }
  if (motor_id == 2){
    digitalWrite(Motor_Z_dir2, LOW);
    digitalWrite(Motor_Z_dir1, HIGH); 
  }
}

void stopMotor(uint8_t motor_id) {
  if (motor_id == 0){
    digitalWrite(Motor_X_dir2, LOW);
    digitalWrite(Motor_X_dir1, LOW); 
    return ;
  }
  if (motor_id == 1){
    digitalWrite(Motor_Y_dir2, LOW);
    digitalWrite(Motor_Y_dir1, LOW); 
    return ;
  }
  if (motor_id == 2){
    digitalWrite(Motor_Z_dir2, LOW);
    digitalWrite(Motor_Z_dir1, LOW); 
  }
}

void update_motor(uint8_t motor_id) {
  curTime[motor_id] = micros();  // change to millis?, micros seem to return a 12 bit number...
  dt = curTime[motor_id] - prevTime[motor_id];

  if (dt == 0) { 
    // function called to soon, avoid division by 0 error or that we have not yet moved
    return;
  }
  prevTime[motor_id] = curTime[motor_id];

  if (dt < 0) {
    // overflow of clock, easiest way to deal with is to just let it pass this time
    return;
  }

  dist = motorPositions[motor_id] - motorPreviousPositions[motor_id];
  motorPreviousPositions[motor_id] = motorPositions[motor_id];
  motorSpeeds[motor_id] = float(dist) / float(dt);

  if (motorSpeeds[motor_id] < 0) {
    motorSpeeds[motor_id] = -motorSpeeds[motor_id];
  }
  
  // TODO add a parameter in in_buffer that tells us if we are doing a move to target on the portenta or we are doing a move at constant speed.
  if (inBuffer[42]==0){
    /*
    By default this is the way the motors are moved, i.e move to target with a target position, if you want to use other moves then set inBuffer[42] = 1
    */
    long dist_diff = 0;
    if (motor_id==0){
      dist_diff = motorPositions[motor_id] - targetPositions[motor_id]; // y goes in the correct direction but does not stop
    }
    else{//} if(motor_id==1){
      dist_diff = targetPositions[motor_id] - motorPositions[motor_id]; // y goes in the correct direction but does not stop
    }

    // (a > b) ? a : b; // max 
    
    if (dist_diff > MOVE_THRESHOLD){// TODO test reducing the movement lim to make smaller moves possible.
      // TODO add adaptive speed here and a max speed parameter to be set in GUI. Perhaps use the same as motor target speed.
      long speed_ = dist_diff*30+340; // *20 + 340
      motorTargetSpeeds[motor_id] = (speed_ < max_speed) ? speed_ : max_speed;
    }
    else if(dist_diff < -MOVE_THRESHOLD){
      long speed_ = dist_diff*30-340;
      motorTargetSpeeds[motor_id] = (speed_ > -max_speed) ? speed_ : -max_speed;
    }
    else{
      motorTargetSpeeds[motor_id] = 0;
    }
  }
  
  // Set move direction
  updateMoveDirection(motor_id);

  update_PID_speed(motor_id, dt);
}

void updateMoveDirection(uint8_t motor_id) {
  if (motorTargetSpeeds[motor_id] == 0) {
    // TODO carefully consider if this should be here or somewhere else.
    stopMotor(motor_id);
    dutyCycle[motor_id] = 0;
    setPWM_DCPercentage_manual(motor_pwm[motor_id], motorPWMPins[motor_id], dutyCycle[motor_id]);
    motorMoveDirections[motor_id] = 0;
    return;
  }
  if (motorTargetSpeeds[motor_id] > 0) {
    moveForward(motor_id);
    motorMoveDirections[motor_id] = 1;
  } else {
    moveBackward(motor_id);
    motorMoveDirections[motor_id] = -1;
  }
}

// Updated to include the I an D parameters using ChatGPT
void update_PID_speed(uint8_t motor_id, uint8_t dt) {
  double speedTarget = 0;
  if (motorTargetSpeeds[motor_id] > 0) {
    speedTarget = double(motorTargetSpeeds[motor_id]) / SPEED_SCALE;
  } else {
    speedTarget = -double(motorTargetSpeeds[motor_id]) / SPEED_SCALE;
  }
  ds = motorSpeeds[motor_id] - speedTarget;

  // Integral
  integral_errors[motor_id] += ds * dt;

  // Derivative
  double derivative = (ds - previous_errors[motor_id]) / dt;
  previous_errors[motor_id] = ds;

  // PID Calculation
  dutyCycle[motor_id] = dutyCycle[motor_id] - (K_P * ds + K_I * integral_errors[motor_id] + K_D * derivative);

  if (dutyCycle[motor_id] < DUTY_CYCLE_MIN) {
    dutyCycle[motor_id] = DUTY_CYCLE_MIN;
  }
  if (dutyCycle[motor_id] > DUTY_CYCLE_MAX) {
    dutyCycle[motor_id] = DUTY_CYCLE_MAX;
  }

  setPWM_DCPercentage_manual(motor_pwm[motor_id], motorPWMPins[motor_id], dutyCycle[motor_id]);
}

void updateMotorControl(uint8_t motor_id, uint8_t dt) {
    double positionError = targetPositions[motor_id] - motorPositions[motor_id]; // Position error

    // Check if this is needed
    if (fabs(positionError)<3){
        stopMotor(motor_id);
        return;
    }
    
    integralError[motor_id] += positionError * dt; // Accumulate integral error
    double derivative = (positionError - lastError[motor_id]) / dt; // Calculate derivative of error
    lastError[motor_id] = positionError; // Update last error

    // PID output for speed control
    double outputSpeed = K_P_position * positionError + K_D_position * derivative; //+ K_I * integralError[motor_id] ;

    // Convert PID output speed to a PWM duty cycle and constrain it
    motorTargetSpeedsPosition[motor_id] = constrain(outputSpeed, -100.0, 100.0); // Assuming outputSpeed is ±100 max
    dutyCycle[motor_id] = mapSpeed2DutyCycle(fabs(motorTargetSpeedsPosition[motor_id]), 0, 100, DUTY_CYCLE_MIN, DUTY_CYCLE_MAX); // Map speed to PWM duty cycle

    // Update motor direction based on PID output
    if (motorTargetSpeedsPosition[motor_id] > 0) {
        moveForward(motor_id);
    } else if (motorTargetSpeedsPosition[motor_id] < 0) {
        moveBackward(motor_id);
    } else {
        stopMotor(motor_id);
    }
    // Apply the duty cycle to the motor
    //setPWM_DCPercentage_manual(motor_pwm[motor_id], motorPWMPins[motor_id], dutyCycle[motor_id]);
}

// Utility function to map values (similar to Arduino map function)
long mapSpeed2DutyCycle(long x, long in_min, long in_max, long out_min, long out_max) {
  return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min;
}





/*
void update_PID_speed(uint8_t motor_id) {
  double speedTarget = 0;
  if (motorTargetSpeeds[motor_id] > 0) {
    // error in speed?
    speedTarget = double(motorTargetSpeeds[motor_id]) / SPEED_SCALE;
  } else {
    // Is this code ever reached?
    speedTarget = -double(motorTargetSpeeds[motor_id]) / SPEED_SCALE;
  }
  ds = motorSpeeds[motor_id] - speedTarget;
  dutyCycle[motor_id] = dutyCycle[motor_id] - ds * k_p; // todo add also I and D in the PID algorithm.

  if (dutyCycle[motor_id] < DUTY_CYCLE_MIN) {
    dutyCycle[motor_id] = DUTY_CYCLE_MIN;
  }
  if (dutyCycle[motor_id] > DUTY_CYCLE_MAX) {
    dutyCycle[motor_id] = DUTY_CYCLE_MAX;
  }

  setPWM_DCPercentage_manual(motor_pwm[motor_id], motorPWMPins[motor_id], dutyCycle[motor_id]);
}
/*