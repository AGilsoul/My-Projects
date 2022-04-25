//
// Created by agils on 4/24/2022.
//

#pragma once

#include <vector>
#include <iostream>
#include <numeric>
#include <algorithm>
#include <bits/stdc++.h>

using std::vector;
using std::cout;
using std::endl;
using std::ostream;
using std::string;
using std::iota;
using std::begin;
using std::end;

class polynomial {

public:
    // default constructor
    polynomial() {}
    // constructor
    polynomial(vector<double> coefs, vector<int> powers, string variable) {
        try {
            if (coefs.size() != powers.size()) { throw std::runtime_error("Incompatible vector sizes"); }
            for (int i = 1; i < powers.size(); i++) {
                if (powers[i] <= powers[i-1]) { throw std::runtime_error("Powers not in ascending order"); }
            }
            int len = powers.back() - powers.front();
            vector<double> tempCoefs(len + 1, 0);
            vector<int> tempPowers(len + 1);
            iota (begin(tempPowers), end(tempPowers), powers.front());
            for (int i = 0; i < coefs.size(); i++) {
                int powerIndex = find(tempPowers.begin(), tempPowers.end(), powers[i]) - tempPowers.begin();
                tempCoefs[powerIndex] = coefs[i];
            }
            this->coefs = tempCoefs;
            this->powers = tempPowers;
            this->order = powers.back();
            this->variable = variable;
        }
        catch (std::runtime_error e) {
            cout << e.what() << endl;
        }
    }

    // overload + operator
    polynomial operator+(polynomial& p2) {
        if (polyComp(p2)) {
            // find min and max orders of polynomials
            int maxOrder = std::max(this->powers.back(), p2.powers.back());
            int minOrder = std::min(this->powers.front(), p2.powers.front());
            // create empty vector for new coefficients
            vector<double> newCoefs(maxOrder - minOrder + 1, 0);
            // create ascending vector for new powers
            vector<int> newPowers(maxOrder - minOrder + 1);
            iota (begin(newPowers), end(newPowers), minOrder);
            // add elements of first vector
            for (int i = 0; i < this->coefs.size(); i++) {
                newCoefs[this->powers[i] - minOrder] += this->coefs[i];
            }
            // add elements of second vector
            for (int i = 0; i < p2.coefs.size(); i++) {
                newCoefs[ p2.powers[i] - minOrder] += p2.coefs[i];
            }
            // create polynomial to return
            polynomial sum(newCoefs, newPowers, this->variable);
            sum.clean();
            return sum;
        }
        else {
            return polynomial();
        }
    }

    // overload - operator
    polynomial operator-(polynomial& p2) {
        auto tempPoly = p2;
        if (polyComp(tempPoly)) {
            for (int i = 0; i < tempPoly.coefs.size(); i++) {
                tempPoly.coefs[i] *= -1;
            }
            auto difference = operator+(tempPoly);
            difference.clean();
            return difference;
        }
        else {
            return polynomial();
        }

    }

    // overload * operator
    polynomial operator*(polynomial& p2) {
        if (polyComp(p2)) {
            int newMaxOrder = this->powers.back() + p2.powers.back();
            int newMinOrder = this->powers.front() + p2.powers.front();
            vector<int> newPowers(newMaxOrder - newMinOrder + 1);
            vector<double> newCoefs(newMaxOrder - newMinOrder + 1, 0);
            iota (begin(newPowers), end(newPowers), newMinOrder);

            for (int p1Count = 0; p1Count < this->powers.size(); p1Count++) {
                for (int p2Count = 0; p2Count < p2.powers.size(); p2Count++) {
                    newCoefs[this->powers[p1Count] + p2.powers[p2Count] - newMinOrder] += this->coefs[p1Count] * p2.coefs[p2Count];
                }
            }
            polynomial product(newCoefs, newPowers, this->variable);
            product.clean();
            return product;
        }
        else {
            return polynomial();
        }
    }

    void clean() {
        for (int i = 0; i < this->powers.size(); i++) {
            if (this->coefs[i] == 0) {
                this->powers.erase(this->powers.begin() + i);
                this->coefs.erase(this->coefs.begin() + i);
                i--;
            }
        }
        order = this->powers.back();
    }

    // function to check if two polynomials have the same variable
    bool polyComp(const polynomial& poly) {
        try {
            if (this->variable == poly.variable) {
                return true;
            }
            else {
                throw std::runtime_error("Incompatible variables");
            }
        }
        catch(std::runtime_error e) {
            cout << e.what() << endl;
            return false;
        }
    }

    // overload << operator friend function to print polynomial to console
    friend ostream& operator<<(ostream& out, const polynomial& poly) {
        out << "(";
        bool firstDone = false;
        double tempCoef;
        for (int c = 0; c < poly.coefs.size(); c++) {
            tempCoef = poly.coefs[c];
            if (poly.coefs[c] != 0) {
                if (c != 0 && firstDone) {
                    if (poly.coefs[c] > 0) {
                        out << "+ ";
                    }
                    else {
                        out << "- ";
                        tempCoef *= -1;
                    }
                }
                if (poly.powers[c] == 0) {
                    out << tempCoef;
                }
                else if (poly.powers[c] == 1) {
                    out << tempCoef << poly.variable;
                }
                else {
                    out << tempCoef << poly.variable << "^" << poly.powers[c];
                }
                if (c != poly.coefs.size() - 1) {
                    out << " ";
                }
                firstDone = true;
            }
        }
        out << ")";
        return out;
    }

private:
    vector<double> coefs;
    vector<int> powers;
    int order;
    string variable;
};
