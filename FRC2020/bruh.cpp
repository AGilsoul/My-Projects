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

using namespace std;


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


//teleop shooting method
void Robot::teleShoot() {
  timeshootstart = timer.GetFPGATimestamp();
  findTarget();

  targetTheta = mountingAngle + table->GetNumber("ty", 0.0);
  targetDistance = (98.25 - mountHeight) / tan(targetTheta);
  plexiAngle = angleMax - (targetTheta / 90) * (angleMax - angleRest);
  
  //Angles the plexiglass according to target distance
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
  
  //Once the flywheel for the launcher has spun up, the conveyor will start moving the balls
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

//autonomous shooting method
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

//Targeting method for autonomous
//if the target is found, starts autoShoot method
void Robot::startAutoTarget() {
  if (findTarget()) {
    autoShoot();
  }
}

//Targeting method for teleop
//if the target is found, starts teleShoot method
void Robot::startTeleTarget() {
  if (findTarget()) {
    teleShoot();
  }
}

//method for finding the target
//turns the robot to face the target until within tolerance
bool Robot::findTarget() {
  //if a target is detected
  validTarget = table->GetNumber("tv",0.0);
  if (validTarget == 1 && !doneshoot) {
    //gets horizontal offset
    targetOH = table->GetNumber("tx",0.0);
    
    if (abs(targetOH) <= tolerance) {
      spinnyBoi.moovmint(0, 0, 0, false);
      return true;
    }
    else if (joystick->GetRawAxis(3) < 0.3){
      spinnyBoi.moovmint(0, 0, 0, false);
    }
    else {
      //turns the robot towards the target using the PID controller
      spinnyBoi.moovmint(0, 0, -1 * (limeControl->Calculate(table->GetNumber("tx", 0.0), limeSetPoint)), true);
    }
  }
  return false;
}

//starts autonomous shooting methods
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


//teleop methods, mostly calling functions if certain buttons are pressed
void Robot::TeleopPeriodic(){
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
  
}


void Robot::TestPeriodic() {
}


#ifndef RUNNING_FRC_TESTS
int main() { return frc::StartRobot<Robot>(); }
#endif
