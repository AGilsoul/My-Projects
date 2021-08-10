#pragma once

#include <iostream>
#include <memory>
#include <vector>
#include <stack>
using namespace std;


template <typename Type>
class Graph;

template <typename Type>
ostream& operator << (ostream & out, const Graph<Type> &g);

template <typename Type>
class Graph {
    private:
        //TODO Add private variables here
        vector<Type> vertices;
        vector<vector<Type>> edges;
        vector<Type> getPath(vector<Type>, Type, Type);

    public:
        Graph();
        void addVertex(Type vertex);
        void addEdge(Type, Type);
        int getVertexPos(Type item);
        int getNumVertices()const;
        bool isEdge(Type, Type);
        friend ostream& operator << <>(ostream & out, const Graph<Type> &g);
        vector<Type>getPath(Type, Type);
};



/*********************************************
* Constructs an empty graph
*
*********************************************/
template<typename Type>
Graph<Type>::Graph() {

}




/*********************************************
* Adds a Vertex to the Graphs. Note that the vertex may not be an int value
*********************************************/
template <typename Type>
void Graph<Type>::addVertex(Type vertex) {
    vertices.push_back(vertex);
    vector<Type> lst;
    edges.push_back(lst);
}

/*********************************************
* Returns the current number of vertices
*
*********************************************/
template<typename Type>
int Graph<Type>::getNumVertices()const {
	return vertices.size();
}



/*********************************************
* Returns the position in the verticies list where the given vertex is located, -1 if not found.
*
*********************************************/
template <typename Type>
int Graph<Type>::getVertexPos(Type item) {
	for (unsigned int i = 0; i < vertices.size(); i++) {
        if (item == vertices[i]) {
            return i;
        }
	}
	return -1; //return negative one
}//End findVertexPos

/*********************************************
* Adds an edge going in the direction of source going to target
*
*********************************************/
template <typename Type>
void Graph<Type>::addEdge(Type source, Type target) {
    int srcPos = getVertexPos(source);
    if (srcPos < 0) {
        throw runtime_error("Vertex not found");
    }
    edges[srcPos].push_back(target);
}

template <typename Type>
bool Graph<Type>::isEdge(Type source, Type dest) {
	int srcPos = getVertexPos(source);
    if (srcPos < 0) {
        throw runtime_error("Vertex not found");
    }
    for (unsigned int i = 0; i < edges[srcPos].size(); i++) {
        if (edges[srcPos][i] == dest) {
            return true;
        }
    }
	return false;
}



/*********************************************
* Returns a display of the graph in the format
* vertex: edge edge
* Note: This is not a traversal, just output
*********************************************/
template <typename Type>
ostream& operator << (ostream & out, const Graph<Type> &g) {
    for (unsigned int i = 0; i < g.vertices.size(); i++) {
        out << "(";
        out << g.vertices[i] << ":";
        for (unsigned int e = 0; e < g.edges[i].size(); e++) {
            out << " " << g.edges[i][e] ;
        }
        out << ")\n";
    }
	return out;
}

/*
  getPath will return the shortest path from source to dest.
  You are welcome to use any solution not limited to the following, depth first search to traverse
  graph to find the solution, breadth first, shortest path first, or any
  other graph algorithm.

  You will return a vector with the solution from the source to the destination.
  IE: The source will be in position 1 the destination is in the last position of the solution, and each node in between
  are the verticies it will travel to get to the destination.  There will not be any
  other verticies in the list.
*/
template <typename Type>
vector<Type> Graph<Type>::getPath(Type source, Type dest) {
	vector<Type> solution;
	//creates an empty vector and sends it to the recursive method
	auto sol = getPath(solution, source, dest);
	return sol;
}

template <typename Type>
vector<Type> Graph<Type>::getPath(vector<Type> solution, Type current, Type dest) {
    int curPos = getVertexPos(current);
    solution.push_back(current);

    //base case: if the last value is the destination, return the solution
    if (solution[solution.size() - 1] == dest) {
        return solution;
    }
	//recursive cases
	vector<Type> tempSol;
	for (unsigned int i = 0; i < edges[curPos].size(); i++) {
        //whether the point has been visited check
        bool been = false;
	    for (unsigned int e = 0; e < solution.size(); e++) {
            if (edges[curPos][i] == solution[e]) {
                been = true;
            }
	    }
        //if the point hasn't been visited
        if (!been) {
            //the new temporary path, if it is shorter, it is returned
            auto temp = getPath(solution, edges[curPos][i], dest);
            if (tempSol.size() == 0) {
                tempSol = temp;
            }
            else if (temp.size() < tempSol.size() && temp.size() != 0) {
                tempSol = temp;
            }
        }
	}
	return tempSol;
}

