/*----------------------------------------------------------------------------*/
/* Copyright (c) 2017-2018 FIRST. All Rights Reserved.                        */
/* Open Source Software - may be modified and shared by FRC teams. The code   */
/* must be accompanied by the FIRST BSD license file in the root directory of */
/* the project.                                                               */
/*----------------------------------------------------------------------------*/

#pragma once

#include <string>

#include <frc/TimedRobot.h>
#include <frc/smartdashboard/SendableChooser.h>

//This file is for setting variables that are used a lot in the bruh.cpp file
//makes it so that we only have to change one variable instead of like 5

class Robot : public frc::TimedRobot {
 public:
  void RobotInit() override;
  void RobotPeriodic() override;
  void AutonomousInit() override;
  void AutonomousPeriodic() override;
  void TeleopInit() override;
  void TeleopPeriodic() override;
  void TestPeriodic() override;
 private:
  frc::SendableChooser<std::string> m_chooser;
  const std::string kAutoNameDefault = "Default";
  const std::string kAutoNameCustom = "My Auto";
  std::string m_autoSelected;
  void startAutoTarget();
  void startTeleTarget();
  void autoShoot();
  void teleShoot();
  bool findTarget();
  frc::Joystick *operatorIn;
  frc::Joystick *joystick;
  frc::Talon *intake;
  frc::Relay *lift1;
  frc::Relay *lift2;
  frc::Talon *conveyer;
  frc::Timer timed;
  frc::Talon *flyWheel;
  frc::Talon *anglerMotor;
  cs::UsbCamera intakeCam;
  cs::UsbCamera driveCam;
  frc::AnalogPotentiometer *pot;
  //resting aiming potentiometer value is 0.433
  //max angle is 1.238
  //right bumper down, left bumper up

  //limelight stuff
  std::shared_ptr<NetworkTable> table = nt::NetworkTableInstance::GetDefault().GetTable("limelight");
  int validTarget;
  double targetOH, targetOV, targetA, autoSpeed;
  double tolerance = 1.0;
  const double plexiConst = 1.0;
  const double angleMax = 0.229;
  const double angleRest = 0.097;
  const double mountingAngle = 30.0;
  const double mountHeight = 1.0;
  const double plexiTolerance = 0.005;

  int Button1Pushed = 0;
  int Button2Pushed = 0;
  double timeStart = 0;
  double flywheeltime = 0;
  double x;
  double ballInterval = .8;
  double ballDelay = 4;
  double intakeSpeed = 0.5;
  Debounce::Debouncer *DebouncerA;
  Debounce::Debouncer *DebouncerB;  
  Debounce::Debouncer *DebouncerX;
  Debounce::Debouncer *DebouncerY;
  Debounce::Debouncer *DebouncerT;
  bool firstTime = true;
};
