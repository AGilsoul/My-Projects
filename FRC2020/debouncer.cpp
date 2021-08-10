#include "debouncer.h"
#include "frc/WPILib.h"

using namespace Debounce;

  Debouncer::Debouncer(frc::Joystick *Joystick, int ButtonNum){
      this->Joystick=Joystick;
      this->ButtonNum=ButtonNum;
      this->lastPress=0;
      this->debouncePeriod=0.5;
  }
  Debouncer::Debouncer(frc::Joystick *Joystick, int ButtonNum, double debouncePeriod){
      this->Joystick=Joystick;
      this->ButtonNum=ButtonNum;
      this->lastPress=0;
      this->debouncePeriod=debouncePeriod;
  }
  int Debouncer::get(){
      frc::Timer timer;
      double now = timer.GetFPGATimestamp();
      if(Joystick->GetRawButton(this->ButtonNum)){
          if((now - this->lastPress)>this->debouncePeriod){
              this->lastPress = now;
              return 1;
          }
      }
            return 0;

  }

void Debouncer::setDebouncePeriod(double newPeriod){
    this->debouncePeriod=newPeriod;
}
