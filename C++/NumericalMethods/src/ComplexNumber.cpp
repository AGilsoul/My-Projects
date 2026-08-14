#include "ComplexNumber.h"
#include <math.h>

using namespace std;

namespace NumericalMethods
{
	double ComplexNumber::getModulus()
	{
		return sqrt(pow(Re, 2) + pow(Im, 2));
	}

	double ComplexNumber::getPhase()
	{
		return atan2(Im, Re);
	}

	void ComplexNumber::operator+=(const ComplexNumber& z)
	{
		this->Re += z.Re;
		this->Im += z.Im;
	}

	ComplexNumber ComplexNumber::operator+(const ComplexNumber& z)
	{
		return ComplexNumber(Re + z.Re, Im + z.Im);
	}

	void ComplexNumber::operator-=(const ComplexNumber& z)
	{
		this->Re -= z.Re;
		this->Im -= z.Im;
	}

	ComplexNumber ComplexNumber::operator-(const ComplexNumber& z)
	{
		return ComplexNumber(Re - z.Re, Im - z.Im);
	}

	void ComplexNumber::operator*=(const ComplexNumber& z)
	{
		double a = Re * z.Re - Im * z.Im;
		double b = Re * z.Im + z.Re * Im;
		this->Re = a;
		this->Im = b;
	}

	ComplexNumber ComplexNumber::operator*(const ComplexNumber& z) {
		return ComplexNumber(Re * z.Re - Im * z.Im, Re * z.Im + z.Re * Im);
	}

	void ComplexNumber::operator*=(double x)
	{
		this->Re *= x;
		this->Im *= x;
	}

	ComplexNumber ComplexNumber::operator*(double x)
	{
		return ComplexNumber(Re * x, Im * x);
	}

	void ComplexNumber::operator/=(double x)
	{
		this->Re /= x;
		this->Im /= x;
	}

	ComplexNumber ComplexNumber::operator/(double x)
	{
		return ComplexNumber(Re / x, Im / x);
	}
}

