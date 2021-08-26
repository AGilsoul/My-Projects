#include <iostream>
#include <chrono>
#include <stdlib.h>

#include "Tree.h"

using std::cout;
using std::endl;
using std::cin;


/*
This is a program that generates a binary tree of a
random size, and reshapes it to be balanced
*/

Tree<int> generateBadTree(unsigned int sizeT) {
    Tree<int> tree;
    int i = 1;
    vector<int> temp;
    int var;
    bool found;
    while (temp.size() < sizeT) {
        found = false;
        srand(i * time(NULL));
        var = rand() % sizeT + 1;
        for (unsigned int x = 0; x < temp.size(); x++) {
            if (temp[x] == var) {
                found = true;
            }
        }
        if (!found) {
            tree.insert(var);
            temp.push_back(var);
        }
        i++;
    }
    return tree;
}

int main()
{
    bool playing = true;
    unsigned int sizeT;
    std::string response;
    while (playing) {
        cout << "________________________________________________________________________________________________________________________" << endl;
        cout << "Enter the node count of the tree for demonstration (If < 25, displays trees; Can only store up to 32768 nodes): ";
        cin >> sizeT;
        cout << endl << "Inserting Nodes..." << endl;
        auto tree = generateBadTree(sizeT);
        int treeCount = tree.nodeCount();
        cout << treeCount << " nodes inserted" << endl;
        double firstAvg = tree.getAvgComp();
        cout << endl << "Average # of comparisons before balancing: " << firstAvg << endl << endl;
        if (treeCount < 25) {
            tree.reverseInOrder();
        }
        cout << "Balancing..." << endl;
        tree.selfBalance();
        cout << "Done balancing" << endl << endl;
        if (treeCount < 25) {
            tree.reverseInOrder();
        }

        double secondAvg = tree.getAvgComp();
        cout << "Average # of comparisons after balancing: " << secondAvg << endl << endl;
        cout << "Post balance search has on average " << 100 - (secondAvg / firstAvg * 100) << "% less comparisons" << endl;
        cout << "Try again with a new tree? (Y/N): ";
        cin >> response;
        if (response != "Y" && response != "y") {
            playing = false;
        }
        cout << "________________________________________________________________________________________________________________________" << endl;
    }


    return 0;
}




