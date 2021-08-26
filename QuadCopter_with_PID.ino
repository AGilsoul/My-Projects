#include <Adafruit_Sensor.h>
#include <Adafruit_BNO055.h>
#include <utility/imumaths.h>
#include <Wire.h>
#include <Servo.h>

//radio input pins
#define THR 3
#define AIL 10
#define ELE 6
#define RUD 11

//Electronic Speed Controller output pins
#define EpinBL 2
#define EpinBR 8
#define EpinFL 7
#define EpinFR 4

//PID constants
//X axis: rotation
#define KPX 0.1
#define KIX 0.0
#define KDX 0.0

//Y axis: forwards and back
#define KPY 1.75
#define KIY 0.15
#define KDY 0.075

//Z axis: left and right
#define KPZ 1.5
#define KIZ 0.25
#define KDZ 0.075

//pretty sure I don't need this
//#define BNO055_SAMPLERATE_DELAY_MS (100)

//Sets the desired angles for when quad is level
//X angle is dependent on current orientation
double recAngleX;
//Y increases when angled forward, decreases when backward
double desired_AngleY = 0.8;

//Z increases when angled right, decreases when angled left
double desired_AngleZ = 0.5;

//PID variables
//error of each axis
double xError, yError, zError;
//last error of each axis
double lastXE, lastYE, lastZE;
//integral of each error
double intgX, intgY, intgZ;
//derivative of each error
double deriX, deriY, deriZ;
//final pid value of each axis
double pidX, pidY, pidZ;

//radians to degrees multiplier
const float Radians_Degrees = 180/3.141592654;

//values for controller input
//general power applied to motors
int Throttle;

//double arrays for X and Y and Z angle values
double angles[3];

//Variables for time keeping
double elapsedTime, time, timePrev;

//Servo variables for Electronic Speed Controllers (ESCs) for each motor: Back Left, Back Right, Front Left, Front Right
Servo EscBL;
Servo EscBR;
Servo EscFL;
Servo EscFR;

//double speed variables sent to their corresponding speed controller
double frSpeed;
double brSpeed;
double flSpeed;
double blSpeed;

//creates the object for the accelerometer/magnetometer for getting angles
Adafruit_BNO055 bno = Adafruit_BNO055(55, 0x28);

//creates a new event
sensors_event_t event; 

int iterations = 0;

struct axis {
  axis() {}
  axis(char aN, double p, double i, double d): axisName(aN), P(p), I(i), D(d) { pidError = 0; pidIntegral = 0; pidDerivative = 0; finPID = 0; } 
  char axisName;
  int joystickVal;
  double P;
  double I;
  double D;
  double lastError;
  double pidError;
  double pidIntegral;
  double pidDerivative;
  double finPID;
  float curAngle;
  float orientation;
  double recAngle;
};

  //affects y axis
  axis Aileron;
  
  //affects z axis
  axis Eleron;
  
  //affects x axis
  axis Rudder;
  

void setup() {
  //affects y axis
  Aileron = axis('y', KPY, KIY, KDY);
  Aileron.recAngle = desired_AngleY;
  
  //affects z axis
  //-1, -1, 1, 1
  Eleron = axis('z', KPZ, KIZ, KDZ);
  Eleron.recAngle = desired_AngleZ;
  
  //affects x axis
  //1, -1, -1, 1
  Rudder = axis('x', KPX, KIX, KDX);
  
  
  //Opens a serial port at 115200 baud
  Serial.begin(115200);
  
  //Set up radio input pins for
  pinMode (THR, INPUT);
  pinMode (AIL, INPUT);
  pinMode (ELE, INPUT);
  pinMode (RUD, INPUT);

  //Attaches ESCs to their pins, with PWM min and max at 1000 and 2000 respectively
  EscBL.attach(EpinBL, 1000, 2000);
  EscBR.attach(EpinBR, 1000, 2000);
  EscFL.attach(EpinFL, 1000, 2000);
  EscFR.attach(EpinFR, 1000, 2000);

  //Calibrate ESCs
  writeAllMotors(0);
  writeAllMotors(180);
  writeAllMotors(0);
  
  //Start wire communications with Accelerometer and Gyroscope
  bno.begin();
  bno.setExtCrystalUse(true);

  delay(10000);
  
  //Start keeping track time in milliseconds
  time = millis();
}

void loop() {
  
  updateAngles(iterations);
  printAngles();
  
  //Time keeping 
  timePrev = time;
  time = millis();
  elapsedTime = (time - timePrev) / 1000;

  //gets controller input from each of the joysticks
  Throttle = pulseIn(THR, HIGH);
  Aileron.joystickVal = pulseIn(AIL, HIGH);
  Eleron.joystickVal = pulseIn(ELE, HIGH);
  Rudder.joystickVal = pulseIn(RUD, HIGH);

  //Maps the controller inputs to values to be used with the Throttle and PID values
  Throttle = map(Throttle, 1089, 1875, 0, 180);
  Aileron.joystickVal = map(Aileron.joystickVal, 1089, 1875, -10, 10);
  Eleron.joystickVal = map(Eleron.joystickVal, 1089, 1875, -10, 10);
  Rudder.joystickVal = map(Rudder.joystickVal, 1089, 1875, -10, 10);

  //UNCOMMENT THIS FOR CONTROLLER INPUT
  //Aileron.recAngle = Aileron.joystickVal + desired_AngleY;
  //Eleron.recAngle = Eleron.joystickVal + desired_AngleZ;

  //sets initial speed for each ESC to 0.9 x Throttle input (0, 180)
  flSpeed = (Throttle * 0.9);
  frSpeed = (Throttle * 0.9); 
  blSpeed = (Throttle * 0.9);
  brSpeed = (Throttle * 0.9);

  //checks to see if controller input affects any of the axis, if not
  //then it runs the PID for that axis
  if (Throttle > 5) {
    runPID(&Eleron);
    runPID(&Aileron);
    //runPID(&Rudder);

    /*
    flSpeed += Rudder.finPID;
    frSpeed -= Rudder.finPID;
    blSpeed -= Rudder.finPID;
    brSpeed += Rudder.finPID;
    */
    
    flSpeed -= Aileron.finPID + Eleron.finPID;
    frSpeed -= Aileron.finPID - Eleron.finPID;
    blSpeed += Aileron.finPID + Eleron.finPID;
    brSpeed += Aileron.finPID - Eleron.finPID;
    
  }
  
  
  
  //applies controller input to the speed of each motor
  
  
  //if the throttle is less than 5, all ESCs are sent 0, else they are sent their corresponding speeds
  if (Throttle < 5) {
    writeAllMotors(0);
  }
  else {
    
    EscFL.write(flSpeed); 
    EscFR.write(frSpeed);
    EscBL.write(blSpeed);
    EscBR.write(brSpeed);
    
  }

  //sets the last error for each axis equal to the current error before looping again
  Rudder.lastError = Rudder.pidError;
  Aileron.lastError = Aileron.pidError;
  Eleron.lastError = Eleron.pidError;
  delay(100);
}


//writes all motors to an int speed s
void writeAllMotors(int s) {
  EscFL.write(s);
  EscFR.write(s);
  EscBL.write(s);
  EscBR.write(s);
}


//function for calculating PID value for the x axis (rotation)
void runPID(axis* curAxis) {
  //Gets error of current x angle to the desired x angle in 360 degrees
  //if xError is negative: too far left
  if (curAxis->axisName = 'x') {
    if (curAxis->curAngle - curAxis->recAngle > 180)  {
      curAxis->pidError = (360 - curAxis->curAngle + curAxis->recAngle);
    }
    else if (curAxis->recAngle - curAxis->curAngle > 180) {
      curAxis->pidError = 360 - curAxis->recAngle + curAxis->curAngle;
    }
    else {
      curAxis->pidError = curAxis->recAngle - curAxis->curAngle;
    }
  }
  else {
    curAxis->pidError = curAxis->recAngle - curAxis->curAngle;
  }
  

  
  //Gets integral (cumulative error over time) for x axis
  curAxis->pidIntegral += curAxis->pidError * elapsedTime;

  //gets error rate of change for each axis
  curAxis->pidDerivative = (curAxis->pidError - curAxis->lastError) / elapsedTime;

  //Gets result PID values
  curAxis->finPID = curAxis->P * curAxis->pidError + curAxis->I * curAxis->pidIntegral + curAxis->D * curAxis->pidDerivative;
  //Serial.print(curAxis->axisName);
  //Serial.print(" Axis PID: ");
}


//updates xyz angles
void updateAngles(int &iterations) {
  //gets event for the Adafruit_BNO055 accelerometer/magnetometer
  bno.getEvent(&event);
  
  Rudder.curAngle = event.orientation.x;
  Aileron.curAngle = event.orientation.y;
  Eleron.curAngle = event.orientation.z;

  /*if (iterations < 10) {
    Aileron.recAngle = event.orientation.y;
    Eleron.recAngle = event.orientation.z;
    iterations++;
  }
  */
  
}

void printAngles() {
  //Prints angles for troubleshooting
  Serial.println("\nX angle: ");
  Serial.println(event.orientation.x, 4);
  Serial.println("Y angle: ");
  Serial.println(event.orientation.y, 4);
  Serial.println("Desired Y: ");
  Serial.println(Eleron.recAngle);
  Serial.println("Z angle: ");
  Serial.println(event.orientation.z, 4);
  Serial.println("Desired Z: ");
  Serial.println(Aileron.recAngle);
}
