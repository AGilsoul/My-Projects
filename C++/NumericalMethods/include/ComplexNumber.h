#pragma once

#include <iostream>
#include <iomanip>

using std::ostream;
using std::setw;
using std::endl;

namespace NumericalMethods
{
	/*
	* ComplexNumber Class
	*  - Stores a real and imaginary part
	*/
	class ComplexNumber
	{
	public:
		/*
		* ComplexNumber base constructor
		*/
		ComplexNumber() {}
		/*
		* ComplexNumber overloaded constructor
		* Params:
		*  - a: Real part
		*  - b: Imaginary part
		*/
		ComplexNumber(double a, double b) : Re(a), Im(b) {}
		/*
		* Returns modulus of the number
		* Returns:
		*  - double: calculated modulus (sqrt(a^2 + b^2))
		*/
		double getModulus();
		/*
		* Returns the phase of the number
		* Returns:
		*  - double: calculated phase (atan2(a, b))
		*/
		double getPhase();
		/*
		* Real part getter
		* Returns:
		*  - double: Real part
		*/
		double getRe() { return Re; }
		/*
		* Imaginary part getter
		* Returns:
		*  - double: Imaginary part
		*/
		double getIm() { return Im; }

		void operator+=(const ComplexNumber& z);
		ComplexNumber operator+(const ComplexNumber& z);
		void operator-=(const ComplexNumber& z);
		ComplexNumber operator-(const ComplexNumber& z);
		void operator*=(const ComplexNumber& z);
		ComplexNumber operator*(const ComplexNumber& z);
		void operator*=(double x);
		ComplexNumber operator*(double x);
		void operator/=(double x);
		ComplexNumber operator/(double x);


		friend ostream& operator<<(ostream& out, const ComplexNumber z)
		{
			out << z.Re;
			if (z.Im > 0)
			{
				out << "+" << z.Im << "i";
			}
			else
			{
				out << "-" << abs(z.Im) << "i";
			}
			return out;
		}

	private:
		double Re = 0, Im = 0;
	};
}


