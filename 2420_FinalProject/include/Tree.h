#include <iostream>
#include <memory>
#include <vector>

using std::shared_ptr;
using std::make_shared;
using std::cout;
using std::endl;
using std::vector;
using std::ostream;


template <typename Type>
struct Node {
    Node(Type data): data(data), left(nullptr), right(nullptr) {}
    Type data;
    shared_ptr<Node<Type>> left;
    shared_ptr<Node<Type>> right;
    int comparisons;
};


template <typename Type>
class Tree {
public:
    Tree(): root(nullptr) {}
    Tree(const Tree<Type>&);
    Tree<Type>& operator=(const Tree<Type>&);
    void insert(const Type& item);
    void remove(const Type& item);
    int nodeCount();
    int leavesCount();
    shared_ptr<Node<Type>> find(const Type& item);
    shared_ptr<Node<Type>> findRightMostNode(shared_ptr<Node<Type>> ptr);
    shared_ptr<Node<Type>> findParent(shared_ptr<Node<Type>> ptr);
    void reverseInOrder();
    Type getRoot() { return root->data; }

    void selfBalance();
    int findHeight();
    double getAvgComp();
    vector<shared_ptr<Node<Type>>> inOrder();

private:
    void remove(shared_ptr<Node<Type>> ptr);
    void insert(const Type& item, shared_ptr<Node<Type>> ptr, int nCount);
    int nodeCount(shared_ptr<Node<Type>> ptr);
    int leavesCount(shared_ptr<Node<Type>> ptr);
    shared_ptr<Node<Type>> copyNode(shared_ptr<Node<Type>> ptr);
    shared_ptr<Node<Type>> find(const Type& item, shared_ptr<Node<Type>> ptr);

    void selfBalance(shared_ptr<Node<Type>> ptr);
    int findHeight(shared_ptr<Node<Type>> ptr);
    void inOrder(shared_ptr<Node<Type>> ptr, vector<shared_ptr<Node<Type>>>& curList);
    vector<shared_ptr<Node<Type>>> inOrder(shared_ptr<Node<Type>> ptr);
    void removeSubTree(shared_ptr<Node<Type>> ptr);
    void reverseInOrder(shared_ptr<Node<Type>> ptr, int spaceCount);
    void balancedInsertion(vector<shared_ptr<Node<Type>>> vc);
    void getAvgComp(shared_ptr<Node<Type>> ptr, vector<int>& curList);
    shared_ptr<Node<Type>> root;
};


//Overloaded constructor
template <typename Type>
Tree<Type>::Tree(const Tree<Type>& tree) {
    root = copyNode(tree.root);
}

//Overloaded = operator
template <typename Type>
Tree<Type>& Tree<Type>::operator=(const Tree<Type>& tree) {
    return Tree(*this);
}


//Method to copy a node
template <typename Type>
shared_ptr<Node<Type>> Tree<Type>::copyNode(shared_ptr<Node<Type>> ptr) {
    if (ptr == nullptr) {
        return nullptr;
    }
    auto ptrCopy = make_shared<Node<Type>>(ptr->data);
    ptrCopy->left = copyNode(ptr->left);
    ptrCopy->right = copyNode(ptr->right);
    return ptrCopy;
}


//Method to insert a node with a given value
template <typename Type>
void Tree<Type>::insert(const Type& item) {
    if (root == nullptr) {
        root = make_shared<Node<Type>>(item);
        root->comparisons = 1;
    }
    else {
       insert(item, root, 1);
    }

}

//Recursive method for above method
template <typename Type>
void Tree<Type>::insert(const Type& item, shared_ptr<Node<Type>> ptr, int nCount) {
    nCount++;
    if (item < ptr->data) {
        if (ptr->left == nullptr) {
            ptr->left = make_shared<Node<Type>>(item);
            ptr->left->comparisons = nCount;
        }
        else {
            insert(item, ptr->left, nCount);
        }
    }
    else if (item > ptr->data) {
        if (ptr->right == nullptr) {
            ptr->right = make_shared<Node<Type>>(item);
            ptr->right->comparisons = nCount;
        }
        else {
            insert(item, ptr->right, nCount);
        }
    }
}


//returns the amount of nodes on the tree
template <typename Type>
int Tree<Type>::nodeCount() {
    return nodeCount(root);
}

//Recursive method for above method
template <typename Type>
int Tree<Type>::nodeCount(shared_ptr<Node<Type>> ptr) {
    if (ptr) {
        return 1 + nodeCount(ptr->left) + nodeCount (ptr->right);
    }
    return 0;
}


//returns the amount of leaves on the tree
template <typename Type>
int Tree<Type>::leavesCount() {
    return leavesCount(root);
}

//Recursive method for above method
template <typename Type>
int Tree<Type>::leavesCount(shared_ptr<Node<Type>> ptr) {
    if (ptr) {
        if (ptr->left || ptr->right) {
            return leavesCount(ptr->left) + leavesCount(ptr->right);
        }
        return 1;
    }
    return 0;
}


//starter method to locate a value and return a node
template <typename Type>
shared_ptr<Node<Type>> Tree<Type>::find(const Type& item) {
    return find(item, root);
}

//Recursive method for above method
template <typename Type>
shared_ptr<Node<Type>> Tree<Type>::find(const Type& item, shared_ptr<Node<Type>> ptr) {
    if (ptr == nullptr) {
        return nullptr;
    }
    else if (ptr->data == item) {
        return ptr;
    }
    else if (item >= ptr->data){
        return find(item, ptr->right);
    }
    else {
        return find(item, ptr->left);
    }
}


//Finds the rightmost node of a tree/subtree
//Starts at the root of said tree/subtree
template <typename Type>
shared_ptr<Node<Type>> Tree<Type>::findRightMostNode(shared_ptr<Node<Type>> ptr) {
    if (ptr->right == nullptr) {
        return ptr;
    }
    return findRightMostNode(ptr->right);
}


//Method to find the parent of a given node
template <typename Type>
shared_ptr<Node<Type>> Tree<Type>::findParent(shared_ptr<Node<Type>> ptr) {
    bool found = false;
    auto cur = root;
    if (root->data == ptr->data) {
        return nullptr;
    }
    int nCount = 0;
    while (!found && nCount < nodeCount()) {
        if (ptr->data >= cur->data) {
            if (ptr->data == cur->right->data) {
                return cur;
            }
            else {
                cur = cur->right;
            }
        }
        else {
            if (ptr->data == cur->left->data) {
                return cur;
            }
            else {
                cur = cur->left;
            }
        }
        nCount++;
    }
    return nullptr;
}


//Starter method for removal of nodes
template <typename Type>
void Tree<Type>::remove(const Type& item) {
    remove(find(item));
}

//recursive method for removing nodes
template <typename Type>
void Tree<Type>::remove(shared_ptr<Node<Type>> ptr) {
    auto parent = findParent(ptr);
    if (parent == nullptr) {
        auto temp = findRightMostNode(ptr->left);
        ptr->data = temp->data;
        remove(temp);
        root = ptr;
    }
    else if (ptr->data >= parent->data) {
        if (ptr->left == nullptr) {
            parent->right = ptr->right;
        }
        else if (ptr->right == nullptr) {
            parent->right = ptr->left;
        }
        else {
            auto temp = findRightMostNode(ptr->left);
            ptr->data = temp->data;
            remove(temp);
            parent->right = ptr;
        }
    }
    else {
        if (ptr->left == nullptr) {
            parent->left = ptr->right;
        }
        else if (ptr->right == nullptr) {
            parent->left = ptr->left;
        }
        else {
            auto temp = findRightMostNode(ptr->left);
            ptr->data = temp->data;
            remove(temp);
            parent->left = ptr;
        }
    }
}


//Self Balancing starter method
template<typename Type>
void Tree<Type>::selfBalance() {
    selfBalance(root);
}

//Self Balancing recursive method
//Finds any subtrees that are unbalanced
//Adds all of the subtree's node values to a vector
//Removes the subtree
//Reinserts the nodes so they will form a balanced subree
template<typename Type>
void Tree<Type>::selfBalance(shared_ptr<Node<Type>> ptr) {
    if (ptr) {
        int leftBranch = findHeight(ptr->left);
        int rightBranch = findHeight(ptr->right);
        if (leftBranch > rightBranch + 1 || rightBranch > leftBranch + 1) {
            auto newSub = inOrder(ptr);
            removeSubTree(ptr);
            balancedInsertion(newSub);
        }
        else {
            selfBalance(ptr->right);
            selfBalance(ptr->left);
        }
    }
}


//method to find the height starting at a node
template<typename Type>
int Tree<Type>::findHeight() {
    if (!root) {
        return 0;
    }
    return findHeight(root);
}

//recursive method for above method
template<typename Type>
int Tree<Type>::findHeight(shared_ptr<Node<Type>> ptr) {
    if (!ptr) {
        return 0;
    }
    else {
        int leftHeight = 0;
        int rightHeight = 0;
        leftHeight = 1 + findHeight(ptr->left);
        rightHeight = 1 + findHeight(ptr->right);
        if (rightHeight > leftHeight) {
            return rightHeight;
        }
        else {
            return leftHeight;
        }
    }


}


//method for ordering BST values in order, public
template <typename Type>
vector<shared_ptr<Node<Type>>> Tree<Type>::inOrder() {
    vector<shared_ptr<Node<Type>>> solution;
    inOrder(root, solution);
    return solution;
}

//method for ordering BST values in order, private
template<typename Type>
vector<shared_ptr<Node<Type>>> Tree<Type>::inOrder(shared_ptr<Node<Type>> ptr) {
    vector<shared_ptr<Node<Type>>> solution;
    inOrder(ptr, solution);
    return solution;
}

//recursive method used by the two methods above
template<typename Type>
void Tree<Type>::inOrder(shared_ptr<Node<Type>> ptr, vector<shared_ptr<Node<Type>>>& curList) {
    if (ptr) {
        inOrder(ptr->left, curList);
        curList.push_back(ptr);
        inOrder(ptr->right, curList);
    }

}


//Removes an entire subtree of nodes starting from a specified pointer
template<typename Type>
void Tree<Type>::removeSubTree(shared_ptr<Node<Type>> ptr) {
    if (root->data == ptr->data) {
        root = nullptr;
        return;
    }
    auto temp = findParent(ptr);
    if (temp) {
        if (ptr->data > temp->data) {
            temp->right = nullptr;
        }
        else if (ptr->data < temp->data) {
            temp->left = nullptr;
        }
    }
}

//Recursive method for printing out the tree, with the top of the tree on
//the left of the screen, and the bottom to the right
//with the right branch on top, and left on bottom
template<typename Type>
void Tree<Type>::reverseInOrder() {
    reverseInOrder(root, 8);
}

template<typename Type>
void Tree<Type>::reverseInOrder(shared_ptr<Node<Type>> ptr, int spaceCount) {
    if (ptr) {
        reverseInOrder(ptr->right, spaceCount * 1.5);

        if (ptr->data != root->data) {
            for (int i = 0; i < spaceCount; i++) {
                cout << " ";
            }
        }
        cout << ptr->data << endl << endl;
        reverseInOrder(ptr->left, spaceCount * 1.5);
    }
}


//Inserts data from the unbalanced subtrees so they will be balanced
//when they are reinserted
template<typename Type>
void Tree<Type>::balancedInsertion(vector<shared_ptr<Node<Type>>> vc) {
    if (vc.size() == 1 || vc.size() == 2) {
        for (unsigned int i = 0; i < vc.size(); i++) {
            insert(vc[i]->data);
        }
    }
    else {
        int midPoint = vc.size() / 2;
        insert(vc[midPoint]->data);
        vector<shared_ptr<Node<Type>>> frontV;
        vector<shared_ptr<Node<Type>>> backV;
        for (int i = 0; i < midPoint; i++) {
            frontV.push_back(vc[i]);
        }
        for (unsigned int i = midPoint + 1; i < vc.size(); i++) {
            backV.push_back(vc[i]);
        }
        balancedInsertion(frontV);
        balancedInsertion(backV);
    }
}


//recursive method, using an in-order traversal recursive method
//to find the average comparison required to get to all
//of the nodes within the tree
template<typename Type>
double Tree<Type>::getAvgComp() {
    vector<int> solution;
    getAvgComp(root, solution);
    int total = 0;
    int amount = 0;
    for (unsigned int i = 0; i < solution.size(); i++) {
        if (solution[i] != -1) {
            total += solution[i];
            amount++;
        }
    };
    return total / (double)amount;
}

template<typename Type>
void Tree<Type>::getAvgComp(shared_ptr<Node<Type>> ptr, vector<int>& curList) {
    if (ptr) {
        getAvgComp(ptr->left, curList);
        curList.push_back(ptr->comparisons);
        getAvgComp(ptr->right, curList);
    }
}

