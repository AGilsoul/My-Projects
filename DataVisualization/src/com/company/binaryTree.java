package com.company;
import javax.swing.*;
import java.util.ArrayList;

//Binary tree object
class binaryTree {
    //Constructor
    public binaryTree(Node topNode) {
        this.topNode = topNode;
    }

    //Method to search for nodes within tree
    public Node search(Node pNode, int value) {
        if (value > pNode.getInt()) {
            if (pNode.getRightNode() != null) {
                return search(pNode.getRightNode(), value);
            }
            else {
                return null;
            }
        }
        else if (value < pNode.getInt()) {
            if (pNode.getLeftNode() != null) {
                return search(pNode.getLeftNode(), value);
            }
            else {
                return null;
            }
        }
        else {
            return pNode;
        }
    }

    //Method to add node to tree
    public void addNode(Node pNode, Node added) {
        if (topNode == null) {
            topNode = added;
        }
        else {
            if (added.getInt() > pNode.getInt()) {
                if (pNode.getRightNode() == null) {
                    pNode.setRightNode(added);
                    added.setParent(pNode);
                }
                else {
                    addNode(pNode.getRightNode(), added);
                }
            }
            else if (added.getInt() < pNode.getInt()) {
                if (pNode.getLeftNode() == null) {
                    pNode.setLeftNode(added);
                    added.setParent(pNode);
                }
                else {
                    addNode(pNode.getLeftNode(), added);
                }
            }
            else {
                pNode.setDataCount();
            }
        }
    }

    //Method to remove a node section from the tree
    public void removeNodeSection(JFrame frame, Node pNode) {
        if (pNode.getParent() != null) {
            if (pNode.getInt() > pNode.getParent().getInt()) {
                pNode.getParent().setRightNode(null);
            }
            else {
                pNode.getParent().setLeftNode(null);
            }
        }
        if (pNode.getS()) {
            frame.remove(hBut);
        }
        frame.remove(pNode.getButton());
        frame.revalidate();
        frame.repaint();

        if (pNode.getRightNode() != null) {
            removeNodeSection(frame, pNode.getRightNode());
        }
        if (pNode.getLeftNode() != null) {
            removeNodeSection(frame, pNode.getLeftNode());
        }

    }

    //Prints all of the nodes in the tree
    public void printNodes(Node pNode) {
        if (pNode.getParent() != null) {
            System.out.println("Node Value: " + pNode.getInt() + " Node Count: " + pNode.getDataCount() + " Parent Node: " + pNode.getParent().getInt());
        }
        else {
            System.out.println("Node Value: " + pNode.getInt() + " Node Count: " + pNode.getDataCount() + " Parent Node: NULL");
        }

        if (pNode.getLeftNode() != null) {
            printNodes(pNode.getLeftNode());
        }

        if (pNode.getRightNode() != null) {
            printNodes(pNode.getRightNode());
        }

    }

    //Returns top node on the tree
    public Node getTopNode() {
        return topNode;
    }

    //Returns the search node button
    public JButton getSearchNode() {
        return hBut;
    }

    //Sets the search node button
    public void setSearchNode(JButton newN) {
        hBut = newN;
    }

    //Private variables
    private JButton hBut;
    private Node topNode;
}
