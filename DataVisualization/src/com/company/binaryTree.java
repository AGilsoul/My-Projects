package com.company;
import javax.swing.*;
import java.util.ArrayList;

class binaryTree {
    public binaryTree(Node topNode) {
        this.topNode = topNode;
    }

    public Node search(Node pNode, int value) {
        //System.out.println(pNode.getInt());
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

    public Node getTopNode() {
        return topNode;
    }

    public JButton getSearchNode() {
        return hBut;
    }

    public void setSearchNode(JButton newN) {
        hBut = newN;
    }

    private JButton hBut;
    private Node topNode;
}