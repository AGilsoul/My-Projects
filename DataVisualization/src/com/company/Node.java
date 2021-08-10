package com.company;
import javax.swing.*;

class Node {
    public Node() {
        this.data = 0;
        this.dataCount = 1;
    }

    public Node(int data) {
        this.data = data;
        dataCount = 1;
    }

    public int getInt() {
        return data;
    }

    public Node getRightNode() {
        return rightNode;
    }

    public Node getLeftNode() {
        return leftNode;
    }

    public void setLeftNode(Node leftNode) {
        this.leftNode = leftNode;
    }

    public void setRightNode(Node rightNode) {
        this.rightNode = rightNode;
    }

    public int getDataCount() {
        return dataCount;
    }

    public void setDataCount() {
        this.dataCount++;
    }

    public void chooseDataC(int num) {
        dataCount = num;
    }

    public void setInt(int num) {
        data = num;
    }

    public void setParent(Node pNode) {
        this.pNode = pNode;
    }

    public Node getParent() {
        return pNode;
    }

    public void setCoord(int x, int y){
        xLoc = x;
        yLoc = y;
    }

    public int getX() {
        return xLoc;
    }

    public int getY() {
        return yLoc;
    }

    public void setButton(JButton button) {
        this.button = button;
    }

    public JButton getButton() {
        return button;
    }

    public void setInc(int inc) {
        curInc = inc;
    }

    public int getInc() {
        return curInc;
    }

    public boolean getS() {
        return searched;
    }

    public void setS(boolean s) {
        searched = s;
    }

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