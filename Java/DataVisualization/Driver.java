package com.company;
import javax.swing.*;
import java.awt.*;
import java.awt.geom.*;
import java.util.*;
import java.awt.event.*;


public class Driver {

    public static void main(String[] args) {
        //Creates a new node to be the top node of the tree
        Node myTopNode = new Node(29);
        //Creates a binary tree with the top node
        binaryTree myTree = new binaryTree(myTopNode);
        //All of the soon to be nodes to be inserted into the tree
        int[] intList = {5, 7, 3, 1, 2, 3, 4, 6, 11, 5, 7, 11, 45, 67, 34, 48, 13, 24, 24, 34, 15, 9};
        for (int i = 0; i < intList.length; i++) {
            Node newNode = new Node(intList[i]);
            myTree.addNode(myTopNode, newNode);
        }
        //Creates a JFrame object with the binaryTree
        JFrame frame = makeGui(myTree);
        //Exits the progam when the GUI window is closed
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);

        //Adds tree data to GUI, with set increments
        addData(frame, myTree.getTopNode(), 490, 10, 200);
        //Creates buttons for modifying tree
        JButton searchNode = new JButton("Search For a Node");
        JButton removeNode = new JButton("Remove Node and all Nodes Beneath");
        JButton addNode = new JButton("Add a Node");
        searchNode.setBounds(50, 550, 150, 50);
        removeNode.setBounds(250, 550, 150, 50);
        addNode.setBounds(450, 550, 150, 50);
        frame.add(searchNode);
        frame.add(removeNode);
        frame.add(addNode);
        frame.setVisible(true);

        //Adds click action listeners to each button, and assigns the tree methods to the corresponding buttons
        searchNode.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent e) {
                int searched = Integer.parseInt(JOptionPane.showInputDialog(frame.getContentPane(), "Enter a Node to Search for"));
                dataSearch(frame, myTree, searched);
            }
        });
        removeNode.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent e) {
                int deleted = Integer.parseInt(JOptionPane.showInputDialog(frame.getContentPane(), "Enter a Node to Delete"));
                myTree.removeNodeSection(frame, myTree.search(myTree.getTopNode(), deleted));
            }
        });
        addNode.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent e) {
                int add = Integer.parseInt(JOptionPane.showInputDialog(frame.getContentPane(), "Enter a Node Value to Add"));
                Node newNode = new Node(add);
                myTree.addNode(myTree.getTopNode(), newNode);
                Node newP = newNode.getParent();
                int pInc = newP.getInc();
                if (add > newP.getInt()) {
                    addData(frame, newNode, newP.getX() + pInc + 25, newP.getY() + pInc/2 + 40, pInc/2);
                }
                else if (add < newP.getInt()) {
                    addData(frame, newNode, newP.getX() - pInc - 25, newP.getY() + pInc/2 + 40, pInc/2);
                }
            }
        });

    }

    //Method for creating the JFrame for the binary tree
    public static JFrame makeGui(binaryTree tree){
        JFrame frame = new JFrame("Binary Tree GUI");
        frame.setSize(1000,1000);
        frame.setLayout(null);
        frame.setVisible(true);
        return frame;
    }

    //Recursive method, adds data to GUI by recursing through the binary tree
    public static void addData(JFrame frame, Node cNode, int x, int y, int inc) {
        JButton add = new JButton(Integer.toString(cNode.getInt()));
        add.setBounds(x, y, 50, 50);
        cNode.setCoord(x, y);
        cNode.setInc(inc);
        frame.add(add);
        frame.revalidate();
        frame.repaint();
        cNode.setButton(add);
        if (cNode.getRightNode() != null) {
            addData(frame, cNode.getRightNode(), x + inc + 25, y + inc/2 + 40, inc/2);
        }
        if (cNode.getLeftNode() != null) {
            addData(frame, cNode.getLeftNode(), x - inc - 25, y + inc/2 + 40, inc/2);
        }
    }

    //Searches for data on the binary tree and highlights it on the GUI
    public static void dataSearch(JFrame frame,binaryTree tree, int value) {
        Node found = tree.search(tree.getTopNode(), value);
        found.setS(true);
        if (found != null) {
            int foundX = found.getX();
            int foundY = found.getY();
            if (tree.getSearchNode() != null) {
                frame.remove(tree.getSearchNode());
            }
            JButton newB = new JButton(Integer.toString(found.getInt()));
            newB.setBackground(Color.BLUE);
            newB.setBounds(foundX - 3, foundY - 3, 56, 56);
            frame.add(newB);
            frame.revalidate();
            frame.repaint();
            tree.setSearchNode(newB);
        }
        else {
            System.out.println("Not Found");
        }
    }
}
