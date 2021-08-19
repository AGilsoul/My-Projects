package com.company;
import javax.swing.*;

//Node object
class Node {
    //Default constructor
    public Node() {
        this.data = 0;
        this.dataCount = 1;
    }

    //Overloaded constructor
    public Node(int data) {
        this.data = data;
        dataCount = 1;
    }

    //Returns data stored in node
    public int getInt() {
        return data;
    }

    //Returns child node to the right
    public Node getRightNode() {
        return rightNode;
    }

    //Returns child node to the left
    public Node getLeftNode() {
        return leftNode;
    }

    //Sets left child node
    public void setLeftNode(Node leftNode) {
        this.leftNode = leftNode;
    }

    //Sets right child node
    public void setRightNode(Node rightNode) {
        this.rightNode = rightNode;
    }

    //Returns node position
    public int getDataCount() {
        return dataCount;
    }

    //Increases node position
    public void setDataCount() {
        this.dataCount++;
    }

    //Sets node position
    public void chooseDataC(int num) {
        dataCount = num;
    }

    //Changes data stored in node
    public void setInt(int num) {
        data = num;
    }

    //Changes the parent node
    public void setParent(Node pNode) {
        this.pNode = pNode;
    }

    //Returns the parent node
    public Node getParent() {
        return pNode;
    }

    //Sets node coordinates on GUI
    public void setCoord(int x, int y){
        xLoc = x;
        yLoc = y;
    }

    //Returns x coordinate
    public int getX() {
        return xLoc;
    }

    //Returns y coordinate
    public int getY() {
        return yLoc;
    }

    //Assigns JButton object to node
    public void setButton(JButton button) {
        this.button = button;
    }

    //Retuns JButton object assigned to node
    public JButton getButton() {
        return button;
    }

    //Sets screen position increment for GUI
    public void setInc(int inc) {
        curInc = inc;
    }

    //Returns screen position increment
    public int getInc() {
        return curInc;
    }

    //Returns whether the node has been searched for or not
    public boolean getS() {
        return searched;
    }

    //Changes value of the searched boolean
    public void setS(boolean s) {
        searched = s;
    }

    //Private variables
    private Node pNode;
    private Node leftNode;
    private Node rightNode;
    private int data;
    private int dataCount;
    private int xLoc;
    private int yLoc;
    private int curInc;
    private boolean searched = false;
    private JButton button;
}
