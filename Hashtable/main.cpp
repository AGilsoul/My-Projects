#include <iostream>
#include <ctime>
#include <cstdlib>
#include <memory>
#include <vector>
#include <string>
#include <random>
#include <math.h>
#include <sstream>

using std::shared_ptr;
using std::cout;
using std::cin;
using std::endl;
using std::string;
using std::make_shared;
using std::srand;
using std::stringstream;


//Node object for each word
struct Node {
    shared_ptr<Node> next = nullptr;
    string data;
};


class hashtable {
public:
    //constructor for hashtable
    hashtable() : itemCount(0), tSize(26) {}
    //insertion function for hashtable
    void insert(string newData) {
        auto newNode = make_shared<Node>();
        newNode->data = newData;
        int pos = (int(newNode->data[0]) % 26);
        if (table[pos] == nullptr) {
            table[pos] = newNode;
        }

        else {
            auto curNode = table[pos];
            while (curNode->next != nullptr) {
                curNode = curNode->next;
            }
            curNode->next = newNode;

        }
    }

    //printing table function for hashtable
    void printTable() {
        for (unsigned int i = 0; i < 26;  i++) {
            cout << i << ": ";
            auto curNode = table[i];
            while (curNode != nullptr) {
                cout << curNode->data;
                curNode = curNode->next;
                if (curNode) {
                    cout << " -> ";
                }
            }
            cout << endl;
        }
    }

private:
    shared_ptr<Node> table[26];
    int tSize;
    int itemCount;
};



int main()
{
    //Creates hashtable object
    hashtable hTable;
    //initialize srand for rng
    srand(time(0));
    string numWords;
    cout << "Specify the number of words to be put in the hashtable: ";
    cin >> numWords;
    int numW = stoi(numWords);
    //Creates a number of random words within the specified parameter
    for (int i = 0; i < numW; i++) {
        int wordLen = (rand() % 10) + 1;
        string curWord;
        for (int l = 0; l < wordLen; l++) {
            curWord += char((rand() % 26) + 96);
        }
        //Inserts each random word into the hashtable
        hTable.insert(curWord);

    }
    //Prints the table
    hTable.printTable();

    return 0;
}


