#include "frc/WPILib.h"

namespace Debounce{
    class Debouncer{
        private:
        frc::Joystick *Joystick;
        int ButtonNum; 
        double lastPress;
        double debouncePeriod;

        public:
            Debouncer(frc::Joystick *Joystick, int ButtonNUm);
            Debouncer(frc::Joystick *Joystick, int ButtonNUm, double debouncePeriod);
            int get();
            void setDebouncePeriod(double newPeriod);
    };
}
