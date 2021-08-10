/*----------------------------------------------------------------------------*/
/* Copyright (c) 2017-2018 FIRST. All Rights Reserved.                        */
/* Open Source Software - may be modified and shared by FRC teams. The code   */
/* must be accompanied by the FIRST BSD license file in the root directory of */
/* the project.                                                               */
/*----------------------------------------------------------------------------*/
#include <iostream>
#include <frc/smartdashboard/SmartDashboard.h>
#include "frc/Talon.h"
#include "rev/ColorSensorV3.h"
#include "rev/ColorMatch.h"
#include <frc/DriverStation.h>
#include <map>
#include "Pheonix.h"
#include "frc/WPILib.h"
#include "debouncer.h"
#include <frc/controller/PIDController.h>
#include "networktables/NetworkTable.h"
#include "networktables/NetworkTableInstance.h"
#include "Robot.h" 
#include "frc/smartdashboard/Smartdashboard.h"
rev::ColorMatch m_colorMatcher;
static constexpr frc::Color kBlueTarget = frc::Color(0, 1, .6);
static constexpr frc::Color kGreenTarget = frc::Color(0, 1, 0);
static constexpr frc::Color kRedTarget = frc::Color(1, 0, 0);
static constexpr frc::Color kYellowTarget = frc::Color(.6, 1, 0);


void Robot::RobotInit() {
  m_colorMatcher.AddColorMatch(kBlueTarget);
  m_colorMatcher.AddColorMatch(kGreenTarget);
  m_colorMatcher.AddColorMatch(kRedTarget);
  m_colorMatcher.AddColorMatch(kYellowTarget);
  m_chooser.SetDefaultOption(kAutoNameDefault, kAutoNameDefault);
  m_chooser.AddOption(kAutoNameCustom, kAutoNameCustom);
  frc::SmartDashboard::PutData("Auto Modes", &m_chooser);
  operatorIn = new frc::Joystick(1);
  joystick = new frc::Joystick(0);
  intake = new frc::Talon(8);
  lift1 = new frc::Relay(1, frc::Relay::kBothDirections);
  lift2 = new frc::Relay(0, frc::Relay::kBothDirections);
  conveyer = new frc::Talon(7);
  flyWheel = new frc::Talon(1);
  anglerMotor = new frc::Talon(2);
  DebouncerA = new Debounce::Debouncer(operatorIn, 1);
  DebouncerB = new Debounce::Debouncer(operatorIn, 2);
  DebouncerX = new Debounce::Debouncer(operatorIn, 3);
  DebouncerY = new Debounce::Debouncer(operatorIn, 4);
  DebouncerT = new Debounce::Debouncer(joystick, 1);
  intakeCam = frc::CameraServer::GetInstance()->StartAutomaticCapture(1);
  driveCam = frc::CameraServer::GetInstance()->StartAutomaticCapture(0);
  pot = new frc::AnalogPotentiometer(4);
}


//swerve drive nonsense
using namespace std;
//each wheel is a WheelBro class
class WheelBro{
    private:
        //creates pointers for motors, PID controllers, and encoders
        frc::Talon *angleMotor;
        WPI_TalonSRX *speedMotor; //change to a WPI_TalonSRX for CAN I think
        frc2::PIDController *bruhcontroller;
        frc::AnalogInput *encoeds;

        //max voltage each encoder can return?
        const double MAX_VOLTS = 5;

    public:
        WheelBro(int angleMotor, int speedMotor, int encoder){
            //sets the angle Motor to the specific Talon it is connected to
            this->angleMotor = new frc::Talon(angleMotor);

            //sets the encoder, similar to a potentiometer but doesn't directly return angles
            this->encoeds = new frc::AnalogInput(encoder);

            //sets the wheel motor that is used for controlling speed
            this->speedMotor = new WPI_TalonSRX(speedMotor); //this should be WPI_TalonSRX I think

            //creates a new Proportion, Integral, Derivative controller with three PID values
            this->bruhcontroller = new frc2::PIDController(.8, .2, 0);
        }

        //speed and angle of each wheel are sent here
        void drive(double speed, double angle){
            //sets the speed of the motor
            this->speedMotor->Set(speed);

            //sets angle of the wheel
            //If robot goes backwards change + to -. this->MAX_VOLTS*0.25 is the offset angle (90 degrees, because thats 1/4 of the max.)
            double setpoint = angle*(this->MAX_VOLTS*0.5)+this->MAX_VOLTS*0.23; 
            
            //These statements are for adjusting the angle of the wheels
            if(abs(setpoint-this->encoeds->GetVoltage())<abs(setpoint+this->MAX_VOLTS-this->encoeds->GetVoltage())){
              //this->angleMotor->Set(.2*(setpoint-this->encoeds->GetVoltage()));
              this->angleMotor->Set(this->bruhcontroller->Calculate(this->encoeds->GetVoltage(), setpoint));
            }
            else{ 
              this->angleMotor->Set(this->bruhcontroller->Calculate(this->encoeds->GetVoltage()-MAX_VOLTS, setpoint));
            }
        }
};


class SpinnyBoi{
    private:
        //creates new WheelBro objects for each motor
        WheelBro *backRight;
        WheelBro *backLeft;
        WheelBro *frontRight;
        WheelBro *frontLeft;
        double speed_coef[2] = {.5, 1};
        int turbo = 0;
    public:
        //Length and width of wheelbase
        const double L = 20+11/16;
        const double W = 25+7/16;
        SpinnyBoi(WheelBro *backRight, WheelBro *backLeft, WheelBro *frontRight, WheelBro *frontLeft){
            this->backRight = backRight;
            this->backLeft = backLeft;
            this->frontRight = frontRight;
            this->frontLeft = frontLeft;
        }

        //checks to see if controller input is within range of the deadband (.25)
        //if it is, then it won't return anything, if not, it returns the 1/2 of the input received
        double deadband(double input){
          if (abs(input) <= .25){
            return 0;
        
          }
          else{
            return input*.5;
          }
        }

        //moovmint takes controller input, x1 and y1 from strafing joystick
        //x2 from rotation joystick, and bool butt determines when to speed up
        void moovmint(double x1, double y1, double x2, bool butt){
            //pythagorean theorem of length and width to get diagonal of wheelbase
            double r = sqrt((L * L) + (W * W));

            //if boost button is pressed, adds one to turbo, making the speed the 1st index(2nd position) of the speed_coef array
            if(butt){
              turbo += 1;
              turbo = remainder(turbo, 2);
            }

            //changes y1 to a negative value
            y1 *= -1;

            //double [letter] = strafing value[x1] +/- rotation value[x2] * (length / diagonal)
            double a = x1-x2*(L/r); 
            double b = x1+x2*(L/r);

            //double [letter] = strafing value[y1] +/- rotation value[x2] * (width / diagonal)
            double c = y1-x2*(W/r);
            double d = y1+x2*(W/r);

           //letters from above input values

           //Speed of back right motor is the negative result of using the pythagorean theorem on b and d
            double backRightSpeed = -sqrt((b*b)+(d*d));
            //Speed of back left motor is the result of using the pythagorean theorem on a and d
            double backLeftSpeed = sqrt((a*a)+(d*d));
            //Speed of front right motor is the negative result of using the pythagorean theorem on b and c
            double frontRightSpeed = -sqrt((b*b)+(c*c));
            //Speed of front left motor is the result of using the pythagorean theorem on a and c
            double frontLeftSpeed = sqrt((a*a)+(c*c));

            //angle of back right wheel is the arctangent of b and d divided by pi
            double backRightAngle = atan2(b, d)/M_PI;
            //angle of back left wheel is the arctangent of a and d divided by pi
            double backLeftAngle = atan2(a, d)/M_PI;
            //angle of back right wheel is the arctangent of b and c divided by pi
            double frontRightAngle = atan2(b, c)/M_PI;
            //angle of back right wheel is the arctangent of a and c divided by pi
            double frontLeftAngle = atan2(a, c)/M_PI;

            //Sends speeds, multiplied by the specific value in the speed_coef array that turbo is the position of, and the corresponding angles to the drive method
            this->backRight->drive(backRightSpeed*speed_coef[turbo], backRightAngle);
            this->backLeft->drive(backLeftSpeed*speed_coef[turbo], backLeftAngle);
            this->frontRight->drive(frontRightSpeed*speed_coef[turbo], frontRightAngle);
            this->frontLeft->drive(frontLeftSpeed*speed_coef[turbo], frontLeftAngle);
        }

        //method for tank drive, sets angles all equal to zero, which wheels all facing straight forward
        void taenck(double y1, double y2){
            this->backRight->drive(y2, 0);
            this->backLeft->drive(y1, 0);
            this->frontRight->drive(y2, 0);
            this->frontLeft->drive(y1, 0);
        }
};


//Creates WheelBro objects for each wheel, with their corresponding motor and encoder numbers
WheelBro *backRight = new WheelBro(5, 3, 3); //change these numbers to: swervy rotatey motor id, movey motor id, encoder id, second encoder id
WheelBro *backLeft = new WheelBro(3, 2, 1); //change these numbers to: swervy rotatey motor id, movey motor id, encoder id, second encoder id
WheelBro *frontRight = new WheelBro(4, 4, 2); //change these numbers to: swervy rotatey motor id, movey motor id, encoder id, second encoder id
WheelBro *frontLeft = new WheelBro(0, 1, 0); //change these numbers to: swervy rotatey motor id, movey motor id, encoder id, second encoder id
SpinnyBoi spinnyBoi(backRight, backLeft, frontRight, frontLeft);
//end of swerve drive nonsense


void Robot::RobotPeriodic() {}

//Initialization for the autonomous section
void Robot::AutonomousInit() {
  //m_autoSelected = m_chooser.GetSelected();
  m_autoSelected = frc::SmartDashboard::GetString("Auto Selector",
       kAutoNameDefault);
  std::cout << "Auto selected: " << m_autoSelected << std::endl;

  if (m_autoSelected == kAutoNameCustom) {
    // Custom Auto goes here
  } else {
    // Default Auto goes here
  }
}

//Initialization for the Teleop section(Empty)
void Robot::TeleopInit() {}

string colorbrochangero(string color){
  string colors[] = {"Y", "B", "G", "R"};
  int n = sizeof(colors)/sizeof(colors[0]);
  int ind = std::distance(colors, find(colors, colors+n, color));
  int newind = remainder(ind+2, n);

  while(newind < 0){
    newind += n;
  }
  string out = colors[newind];
  return out;
}
/*
map<frc::Color, string> culir2lettr = {
    {kBlueTarget, "B"},
    {kGreenTarget, "G"},
    {kRedTarget, "R"},
    {kYellowTarget, "Y"}
};
*/
//map<frc::Color, string> culir2lettr; a dictionary would make converting frc colors to strings a lot easier but it didnt want to work
//returns a string based on the color detected
string culir2lettr(frc::Color color){
    if(color==kBlueTarget){
      return "B";
    }
    else if(color==kGreenTarget){
      return "G";
    }
    else if(color==kRedTarget){
      return "R";
    }
    else if(color==kYellowTarget){
      return "Y";
    } //WHY do cpp switch statements only work for ints?
    return "N";
}

//Most of these are variables used for autonomous/teleop periodic
static constexpr auto i2cPort = frc::I2C::Port::kOnboard;
rev::ColorSensorV3 m_colorSensor{i2cPort};
string colorString;
frc::Talon motor(6);
frc::Color detectedColor;
frc::Color matchedColor;
string gameData; 
double confidence;
string goal;
string cletter;
double timestart;
frc::Timer timer;
double rotations = 21.4;
double speed = 500/60;
double rotateTime = rotations*2/speed;
bool culir = false;
//Debounce::Debouncer debouncer(joystick, 10);
bool rotation = false;
bool inconveyor = false;
const double robotspeed = 11.5;
const double robotlength = 20.6875;
const double robotwidth = 25.4375;
const double robotradius = sqrt((robotlength * robotlength) + (robotwidth * robotwidth));
const double timeautoforward = (robotlength+10)/(2*robotspeed);
const double turnangledist = .95*3*1.7*13*robotradius*M_PI/(4*18*10);
const double turntime = .635*turnangledist/robotspeed;
bool mooving = false;
bool doneturn = false;
bool doneshoot = false;
double timeautostart, timemoovstart, timeshootstart, plexiMult, plexiAngle, targetDistance, targetTheta;
double limeSetPoint = 0.0;
bool doneFinding;


//shooting method
//pot max angle: 0.461, min angle: 1.14
//positive value goes to min, negative value goes to max
// 0, 90 -> 1.14, 0.461
void Robot::teleShoot() {
  timeshootstart = timer.GetFPGATimestamp();
  
  findTarget();

  targetTheta = mountingAngle + table->GetNumber("ty", 0.0);
  targetDistance = (98.25 - mountHeight) / tan(targetTheta);
  plexiAngle = angleMax - (targetTheta / 90) * (angleMax - angleRest);
  
  
  while (abs(pot->PIDGet() - plexiAngle) >= plexiTolerance && joystick->GetRawAxis(3) > 0.3) {
    //cout << targetTheta << endl;
    //cout << pot->PIDGet() << endl;
    //cout << angleControl->Calculate(pot->PIDGet(), plexiAngle);
    //anglerMotor->Set(anglePID->calculate(pot->PIDGet(), plexiAngle));
    
    anglerMotor->Set(-10 * (pot->PIDGet() - plexiAngle));
    if (pot->PIDGet() > angleMax || pot->PIDGet() < angleRest) {
      anglerMotor->Set(0);
      //cout << "Out of Range" << endl;
    }
    else {
      //cout << "In Range" << endl;
    }
    
  }
  anglerMotor->Set(0);
  
  
  while (timer.GetFPGATimestamp()-timeshootstart < 10.5 && joystick->GetRawAxis(3) > 0.3){
    if (timer.GetFPGATimestamp()-timeshootstart < 4.5) {
      spinnyBoi.moovmint(0, 0, 0, false);
      flyWheel->Set(-1);
    }
    else {
      conveyer->Set(-.35);
    }
  }
  flyWheel->Set(0);
  conveyer->Set(0);
  //doneshoot = true;
  spinnyBoi.moovmint(0, 0, 0, false);
}

void Robot::autoShoot() {
  doneshoot = true;
  timeshootstart = timer.GetFPGATimestamp();
  while (timer.GetFPGATimestamp()-timeshootstart < 10.5){
    spinnyBoi.moovmint(0, 0, 0, false);
    if (timer.GetFPGATimestamp()-timeshootstart < 4.5) {
      flyWheel->Set(-1);
    }
    else {
      conveyer->Set(-.35);
    }  
  }
  flyWheel->Set(0);
  conveyer->Set(0);
  //doneshoot = true;
  spinnyBoi.moovmint(0, 0, 0, false);
}

//PID for the limelight
frc2::PIDController *limeControl = new frc2::PIDController(0.035, 0.0, 0.8);

//Targeting method
void Robot::startAutoTarget() {
  if (findTarget()) {
    autoShoot();
  }
  
}


void Robot::startTeleTarget() {
  if (findTarget()) {
    teleShoot();
  }
}

bool Robot::findTarget() {
  validTarget = table->GetNumber("tv",0.0);
  if (validTarget == 1 && !doneshoot) {
    targetOH = table->GetNumber("tx",0.0);
    //Experimental Limelight PID
    
    if (abs(targetOH) <= tolerance) {
      spinnyBoi.moovmint(0, 0, 0, false);
      return true;
    }
    else if (joystick->GetRawAxis(3) < 0.3){
      spinnyBoi.moovmint(0, 0, 0, false);
    }
    else {
      spinnyBoi.moovmint(0, 0, -1 * (limeControl->Calculate(table->GetNumber("tx", 0.0), limeSetPoint)), true);
    }
  }
  return false;
}


void Robot::AutonomousPeriodic() {
  if (!mooving) {
    timeautostart = timer.GetFPGATimestamp();
    mooving = true;
  }
  if(!doneshoot){
      targetOH = table->GetNumber("tx",0.0);
      doneFinding = false;
      startAutoTarget();
      //spinnyBoi.moovmint(0, 0, 0, false);
  }
  else {
    //move wherever needed, most likely backwards
  }
}


void Robot::TeleopPeriodic(){
  //cout << pot->PIDGet() << endl;
  spinnyBoi.moovmint(spinnyBoi.deadband(joystick->GetRawAxis(1)), spinnyBoi.deadband(joystick->GetRawAxis(0)), spinnyBoi.deadband(joystick->GetRawAxis(4)), DebouncerT->get());
  //spinnyBoi.taenck(joystick->GetRawAxis(0), joystick->GetRawAxis(5));
  //merged operator code begins here

  //section for the intake, checks to see if certain buttons are pressed
  if(DebouncerA->get()){
      Button1Pushed = remainder((Button1Pushed + 1), 2); 
    }
  if(DebouncerB->get()){
    Button2Pushed = remainder((Button2Pushed + 1), 2);
  }

  //if button A is pushed, set the intake to take in balls?
 if(Button1Pushed == 1) {
   intake->Set(-intakeSpeed);
 }

 //else if button b is pushed, set the intake to push out balls?
  else if(Button2Pushed == 1) {
   intake->Set(intakeSpeed);
 }

 //if neither button is pushed, have conveyor do nothing?
 else{
  intake->Set(0);
 }

 //lift stuff
 if(operatorIn->GetRawButton(7)){
   lift1->Set(frc::Relay::kOn);
   lift2->Set(frc::Relay::kOn);
   lift1->Set(frc::Relay::kForward);
   lift2->Set(frc::Relay::kForward);   
 }
  else if(operatorIn->GetRawButton(8)){
   lift1->Set(frc::Relay::kOn);
   lift2->Set(frc::Relay::kOn);
   lift1->Set(frc::Relay::kReverse);
   lift2->Set(frc::Relay::kReverse);   
 }
 else{
   lift1->Set(frc::Relay::kOff);
   lift2->Set(frc::Relay::kOff);     
 }

 //coneveyer forward
  if(conveyer->Get() == -.5 && timed.GetFPGATimestamp() - timeStart > ballInterval){
   conveyer->Set(0);
   inconveyor = false;
 }
 if(DebouncerX->get()){
   conveyer->Set(-0.5);
   timeStart = timed.GetFPGATimestamp();
   inconveyor = true;
 }
 //conveyer backwards
 if(conveyer->Get() == .5 && timed.GetFPGATimestamp() - timeStart > ballInterval){
   conveyer->Set(0);
   inconveyor = false;
 }
 if(DebouncerY->get()){
   conveyer->Set(0.5);
   timeStart = timed.GetFPGATimestamp();
   inconveyor = true;
 }

 //launcher stuff (shoutouts to Mr. Reeves)
 //if the right trigger is pushed so that a value of more than .3 is returned
 //sets the flywheel to speed
  if (joystick->GetRawAxis(3) > 0.3) {
    while (joystick->GetRawAxis(3) > 0.3) {
      startTeleTarget();
    }
    
  }
  //else sets the flywheel to 0
  else{
    flyWheel->Set(0);
  }

  //plexiglass angling for launcher   
  //if one of the bumpers is pressed, changes the plexiglass angle
  //makes sure the plexiglass isn't angled too far forward or back
  if (joystick->GetRawButton(6) == 1){
    anglerMotor->Set(.3);
  }else if (joystick->GetRawButton(5) == 1 & pot->PIDGet() < 1.238){
    anglerMotor->Set(-.3);
  }else if (joystick->GetRawButton(5) == 1 & pot->PIDGet() >= 1.238){
    anglerMotor->Set(0);
  }else {
    anglerMotor->Set(0);
  }
  
  //Stuff for the color disk, never ended up using this
  detectedColor = m_colorSensor.GetColor();
  confidence = 0.0;
  colorString = culir2lettr(m_colorMatcher.MatchClosestColor(detectedColor, confidence));
  gameData = frc::DriverStation::GetInstance().GetGameSpecificMessage();
  if(operatorIn->GetRawButton(10)&&!rotation&&!culir){
    motor.Set(1);
    timestart = timer.GetFPGATimestamp();
    rotation = true;
  }
  if(rotation&&timer.GetFPGATimestamp()-timestart>rotateTime&&!culir){
    motor.Set(0);
    rotation = false;
  }
  if(gameData.length() > 0){
    goal = colorbrochangero(string(1, gameData[0]));
    detectedColor = m_colorSensor.GetColor();
    matchedColor = m_colorMatcher.MatchClosestColor(detectedColor, confidence);
    cletter = culir2lettr(matchedColor);
    if(operatorIn->GetRawButton(9)&&cletter!=goal&&!rotation){
      motor.Set(1);
      culir = true;
    }
    else if(!rotation){
      motor.Set(0);
      culir = false;
    }
  }
}


void Robot::TestPeriodic() {
  cout << table->GetNumber("tx",0.0) << endl;
}


#ifndef RUNNING_FRC_TESTS
int main() { return frc::StartRobot<Robot>(); }
#endif