#include <iostream>
#include <vector>
#include <utility>

#include <algorithm>
#include <cmath>

#include <string>
#include <fstream>
#include <sstream>
#include <iomanip>

#include <chrono>
using namespace std;
using namespace std::chrono;

class Point {
public:
  double x;
  double y;
  string id;
  Point() {}
  Point(double x, double y, string id = "__") {
    this->x = x;
    this->y = y;
    this->id = id;
  }
};

class Edge {
public:
  Point a;
  Point b;
  string id;
  Edge() {}
  Edge(Point a, Point b, string id = "__") {
    this->a = a;
    this->b = b;
    this->id = id;
  }
};



double pointPointDistance(Point a, Point b) {
  double dx = b.x - a.x;
  double dy = b.y - a.y;
  return sqrt(dx * dx + dy * dy);
}

double pointEdgeDistance(Point p, Edge e) {
  Point a = e.a;
  Point b = e.b;

  double abx = b.x - a.x;
  double aby = b.y - a.y;

  double apx = p.x - a.x;
  double apy = p.y - a.y;

  double ab_ab = abx * abx + aby * aby;
  double ap_ab = apx * abx + apy * aby;
  double t = ap_ab / ab_ab;

  t = max(0.0, min(1.0, t));

  double projx = a.x + t * abx;
  double projy = a.y + t * aby;

  double dx = p.x - projx;
  double dy = p.y - projy;
  return sqrt(dx * dx + dy * dy);
}



class Ball {
public:
  Point centroid;
  double radius;
  Edge edge;
  
  Ball(Point centroid, double radius, Edge edge = Edge()) {
    this->centroid = centroid;
    this->radius = radius;
    this->edge = edge;
  }
  Ball() {}
};

class Node {
public:
  vector<Node> children;
  Ball ball;

  Node() {
    ball = Ball();
  }

  Node(Ball b) {
    ball = b;
  }

  bool isLeaf() const {
    return children.empty();
  }
};

class BallStarTree {
public:
  Node root;

  //! Constructor desde un vector de aristas
  BallStarTree(vector<Edge> edges) {
    vector<Ball> leaves = ballsFromEdges(edges);

    edges.clear();

    root = buildTree(leaves);

    cout << "FINISHED BUILDING" << endl;
  }

  //! Constructor desde nombre de 2 archivos
  BallStarTree(string filenameNodes, string filenameEdges) {
    vector<Point> points = readCSVPoints(filenameNodes);
    vector<Edge> edges = readCSVEdges(filenameEdges, points);

    points.clear();

    vector<Ball> leaves = ballsFromEdges(edges);

    edges.clear();

    root = buildTree(leaves);

    cout << "FINISHED BUILDING THE TREE" << endl;

  }

  //! Buscar la arista mas cercana a un punto
  string nearestEdge(Point p) {
    return nearestEdgeRecursive(p, root, numeric_limits<double>::max(), "__").second;
  }

  //! Guardar la estructura completa en un CSV
  void saveToCSV(string filename) {
    ofstream file(filename);

    file << "nivel,centrox,centroy,radio,x1,y1,x2,y2\n";

    saveNodeToCSV(root, file, 0);

    file.close();
  }

private:
  //! Obtener los puntos desde el CSV de puntos
  vector<Point> readCSVPoints(string filename) {
    vector<Point> points;
    ifstream file(filename);
    string line, cell;
    
    getline(file, line);
    
    while (getline(file, line)) {
      stringstream lineStream(line);
      vector<string> cells;
      
      while (getline(lineStream, cell, ',')) {
        cells.push_back(cell);
      }
      
      if(cells.size() >= 3) {
        string id = cells[0];
        double x = stod(cells[1]);
        double y = stod(cells[2]);

        Point newPoint(x, y, id);

        points.push_back(newPoint);
      }
    }

    return points;
  }

  //! Encontrar los puntos por la ID
  Point findPointById(vector<Point> points, string id) {
    for (int i = 0; i < points.size(); i++) {
      if (points[i].id == id ) {
        return points[i];
      }
    }
    return Point(0, 0);
  }

  //! Obtener las aristas desde los puntos y el CSV de aristas
  vector<Edge> readCSVEdges(string filename, vector<Point> points) {
    ifstream file(filename);
    string line, cell;
    vector<Edge> edges;

    getline(file, line);

    while (getline(file, line)) {
      stringstream lineStream(line);
      vector<string> row;
        
      while (getline(lineStream, cell, ',')) {
        row.push_back(cell);
      }

      if (row.size() < 4) {
        continue;
      }

      string u = row[0];
      string v = row[1];
      string osmid = row[3];

      Point pointU = findPointById(points, u);
      Point pointV = findPointById(points, v);

      if (pointU.id == "__" || pointV.id == "__") {
        continue;
      }

      Edge edge(pointU, pointV, osmid);
      edges.push_back(edge);
    }

    file.close();
    return edges;
  }

  //! Encerrar las aristas en una bola cada una
  vector<Ball> ballsFromEdges(vector<Edge> edges) {
    vector<Ball> balls;

    for(int i = 0; i < edges.size(); i++) {
      double centroid_x = (edges[i].a.x + edges[i].b.x) / 2.0;
      double centroid_y = (edges[i].a.y + edges[i].b.y) / 2.0;
      Point centroid(centroid_x, centroid_y);

      double dx = edges[i].b.x - edges[i].a.x;
      double dy = edges[i].b.y - edges[i].a.y;
      double distance = sqrt(dx * dx + dy * dy);
      double radius = distance / 2.0;

      Ball ball(centroid, radius, edges[i]);
      balls.push_back(ball);
    }

    return balls;
  }



  //! Punto medio desde un vector de puntos
  Point mean(vector<Point> points) {
    double sumX = 0;
    double sumY = 0;
    for (int i = 0; i < points.size(); i++) {
      sumX = sumX + points[i].x;
      sumY = sumY + points[i].y;
    }
    double meanX = sumX / points.size();
    double meanY = sumY / points.size();

    Point result(meanX, meanY);
    return result;
  }

  //! Estandarizar los puntos de un vector de puntos
  vector<Point> standardize(vector<Point> points) {
    Point meanPoint = mean(points);

    vector<Point> standardizedPoints;

    for (int i = 0; i < points.size(); i++) {
      double standardizedX = (points[i].x - meanPoint.x);
      double standardizedY = (points[i].y - meanPoint.y);
      Point standardizedPoint(standardizedX, standardizedY);
      standardizedPoints.push_back(standardizedPoint);
    }
    return standardizedPoints;
  }

  //! Hallar la covarianca de un vector de puntos
  double covariance(vector<Point> points, int cases) {
    double cov = 0;

    Point meanStandarized = mean(points);
    for (int i = 0; i < points.size(); i++) {
      if (cases == 0) {
        cov += (points[i].x - meanStandarized.x) * (points[i].x - meanStandarized.x);
      } 
      else if (cases == 1){
        cov += (points[i].x - meanStandarized.x) * (points[i].y - meanStandarized.y);
      }
      else if (cases == 2) {
        cov += (points[i].y - meanStandarized.y) * (points[i].y - meanStandarized.y);
      }
    }

    return cov / (points.size() - 1);
  }
  
  //! Hallar la matriz de covariancia de un vector de bolas
  vector<vector<double>> covarianceMatrix(vector<Ball> balls) {
    vector<Point> centroids;
    for (int i = 0; i < balls.size(); i++) {
      centroids.push_back(balls[i].centroid);
    }

    vector<Point> stdPoints = standardize(centroids);

    double covXX = covariance(stdPoints, 0);
    double covXY = covariance(stdPoints, 1);
    double covYY = covariance(stdPoints, 2);

    return {
      {covXX, covXY},
      {covXY, covYY}
    };
  }
  
  //! Hallar autovectores y autovalores desde un vector de bolas
  vector<pair<double, vector<double>>> eigen(vector<Ball> balls) {
    vector<vector<double>> matrix = covarianceMatrix(balls);
    vector<pair<double, vector<double>>> result;

    double a = 1.0;
    double b = -(matrix[0][0] + matrix[1][1]);
    double c = (matrix[0][0] * matrix[1][1]) - (matrix[0][1] * matrix[1][0]);
    double discriminant = b * b - 4 * a * c;

    if (discriminant >= 0) {
      double lambda1 = (-b + sqrt(discriminant)) / (2 * a);
      double lambda2 = (-b - sqrt(discriminant)) / (2 * a);

      for (double lambda : {lambda1, lambda2}) {
        vector<double> eigenVector(2);
        
        if (abs(matrix[0][1]) > 0) {
          eigenVector[0] = matrix[0][1];
          eigenVector[1] = lambda - matrix[0][0];
        } 
        else if (abs(matrix[1][0]) > 0) {
          eigenVector[0] = lambda - matrix[1][1];
          eigenVector[1] = matrix[1][0];
        } 
        else {
          if (abs(matrix[0][0] - lambda) < 0) {
            eigenVector[0] = 1;
          }
          else {
            eigenVector[0] = 0;
          }
          if (abs(matrix[1][1] - lambda) < 0) {
            eigenVector[1] = 1;
          }
          else {
            eigenVector[1] = 0;
          }
        }

        double norm = sqrt(eigenVector[0] * eigenVector[0] + eigenVector[1] * eigenVector[1]);
        eigenVector[0] = eigenVector[0] / norm;
        eigenVector[1] = eigenVector[1] / norm;

        if (eigenVector[0] < 0 || (eigenVector[0] == 0 && eigenVector[1] < 0)) {
          eigenVector[0] = -eigenVector[0];
          eigenVector[1] = -eigenVector[1];
        }

        result.push_back({lambda, eigenVector});
      }
    }

    return result;
  }

  //! Ordenar autovectores y autovalores para hallar autovector mas representativo
  void sortEigenPairs(vector<pair<double, vector<double>>> eigenPairs) {
    sort(eigenPairs.begin(), eigenPairs.end(), [](pair<double, vector<double>> a, pair<double, vector<double>> b) {
      return a.first > b.first;
    });
  }

  //! Hallar el componente principal del vector de bolas
  vector<double> principalComponent(vector<Ball> balls) {
    vector<pair<double, vector<double>>> eigenVV = eigen(balls);

    sortEigenPairs(eigenVV);

    //cout << eigenVV[0].second[0] << ' ' << eigenVV[0].second[1] << endl;
    return eigenVV[0].second;
  }



  //! Producto punto de un autovector y un punto para hallar proyección
  double dotProduct(vector<double> vec, Point p) {
    return vec[0] * p.x + vec[1] * p.y;
  }

  //! Función objetivo
  double evaluateObjectiveFunction(int N, int N1, int N2, double tc, double tmin, double tmax, double alpha) {
    double f1 = abs(N2 - N1) / static_cast<double>(N);
    double f2 = (tc - tmin) / (tmax - tmin);
    return f1 + alpha * f2;
  }
  
  //! Encontrar el mejor hyperplano que divide un vector de bolas en un vector
  vector<double> getSplittingHyperplane(vector<double> PC1, vector<Ball> balls, int S = 10, double alpha = 0.5) {
    vector<Point> centroids;
    for (int i = 0; i < balls.size(); i++) {
      centroids.push_back(balls[i].centroid);
    }

    vector<double> projections(centroids.size());
    for (int i = 0; i < centroids.size(); ++i) {
      projections[i] = dotProduct(PC1, centroids[i]);
    }

    double tmin = *min_element(projections.begin(), projections.end());
    double tmax = *max_element(projections.begin(), projections.end());

    double bestTc = tmin;
    double minObjective = numeric_limits<double>::max();

    for (int i = 0; i <= S; ++i) {
      double tc = tmin + i * (tmax - tmin) / S;

      int N1 = 0, N2 = 0;
      for (int j = 0; j < projections.size(); j++) {
        if (projections[j] < tc) {
          N1++;
        } 
        else {
          N2++;
        }
      }

      double objective = evaluateObjectiveFunction(centroids.size(), N1, N2, tc, tmin, tmax, alpha);

      if (objective < minObjective) {
        minObjective = objective;
        bestTc = tc;
      }
    }

    double b = bestTc;
    //cout << PC1[0] << ' ' << PC1[1] << ' ' << -b << endl;
    return {PC1[0], PC1[1], -b};
  }



  //! Dividir un vector de bolas en dos vectores de bolas
  pair<vector<Ball>, vector<Ball>> divideBalls(vector<Ball> balls) {
    vector<Ball> group1, group2;
    vector<double> PC1 = principalComponent(balls);
    
    vector<double> hyperplane = getSplittingHyperplane(PC1, balls);

    double a = hyperplane[0], b = hyperplane[1], c = hyperplane[2];

    for (int i = 0; i < balls.size(); i++) {
      double value = a * balls[i].centroid.x + b * balls[i].centroid.y + c;
      if (value < 0) {
        group1.push_back(balls[i]);
      } 
      else {
        group2.push_back(balls[i]);
      }
    }

    return {group1, group2};
  }
  


  //! Punto mas lejano desde un punto con un vector de puntos
  Point furthestPoint(Point p, vector<Point> centroids) {
    Point currentMaxP;
    double currentMax = 0;
    for (int i = 0; i < centroids.size(); i++) {
      if (pointPointDistance(p, centroids[i]) > currentMax) {
        currentMax = pointPointDistance(p, centroids[i]);
        currentMaxP = centroids[i];
      }
    }
    return currentMaxP;
  }

  //! Division clasica implementado en un ball tree
  pair<vector<Ball>, vector<Ball>> classicDivideBalls(vector<Ball> balls) {
    Point rPoint = balls[0].centroid;
    vector<Point> centroids;
    for (int i = 0; i < balls.size(); i++) {
      centroids.push_back(balls[i].centroid);
    }

    Point fPoint1 = furthestPoint(rPoint, centroids);
    Point fPoint2 = furthestPoint(fPoint1, centroids);
  
    double m = (fPoint2.y - fPoint1.y) / (fPoint2.x - fPoint1.x);
    double b = fPoint1.y - m * fPoint1.x;

    vector<pair<Ball, double>> projections;

    for (int i = 0; i < balls.size(); i++) {
      double proj = (balls[i].centroid.x + m * (balls[i].centroid.y - b)) / (1 + m * m);
      projections.emplace_back(balls[i], proj);
    }

    sort(projections.begin(), projections.end(), [](pair<Ball, double>& a, pair<Ball, double>& b) {
      return a.second < b.second;
    });

    vector<Ball> group1, group2;
    for (int i = 0; i < projections.size(); i++) {
      if (i < projections.size() / 2) {
        group1.push_back(projections[i].first);
      }
      else {
        group2.push_back(projections[i].first);
      }
    }

    return {group1, group2};
  }

  

  //! Construir arbol recursivo
  Node buildTree(vector<Ball> balls, int level = 0) {
    if (balls.size() == 1) {
      return Node(balls[0]);
    }

    pair<vector<Ball>, vector<Ball>> groups = divideBalls(balls);
    
    if (groups.first.size() == 0 || groups.second.size() == 0) {
      //cout << (groups.second.size() == 0 ? groups.first.size() : groups.second.size()) << ' ';
      groups = classicDivideBalls(balls);
    }

    Node node;
    node.ball = createEnclosingBall(balls);

    node.children.push_back(buildTree(groups.first, level + 1));
    node.children.push_back(buildTree(groups.second, level + 1));

    return node;
  }

  //! Encapsular vector de bolas en una sola bola
  Ball createEnclosingBall(vector<Ball> balls) {
    vector<Point> edgePoints;

    Edge defaultEdge;

    for (int i = 0; i < balls.size(); i++) {
      edgePoints.push_back(balls[i].edge.a);
      edgePoints.push_back(balls[i].edge.b);
    }

    Point meanCentroid = mean(edgePoints);

    double maxRadius = 0;
    for (int i = 0; i < balls.size(); i++) {
      if (balls[i].edge.a.id == defaultEdge.a.id && balls[i].edge.b.id == defaultEdge.b.id) {
        double distance = pointPointDistance(meanCentroid, balls[i].centroid);
        maxRadius = max(maxRadius, distance);
      }
      else {
        double distanceA = pointPointDistance(meanCentroid, balls[i].edge.a);
        double distanceB = pointPointDistance(meanCentroid, balls[i].edge.b);
        maxRadius = max(maxRadius, max(distanceA, distanceB));
      }
    }

    return Ball(meanCentroid, maxRadius);
  }



  //! Funcion recursiva de busqueda de arista cercana
  pair<double, string> nearestEdgeRecursive(Point p, const Node& node, double minDist, string minId) const {
    double centerDist = pointPointDistance(p, node.ball.centroid);

    if (centerDist > node.ball.radius + minDist) {
      //cout << "Pruning!" << endl;
      return {minDist, minId};
    }

    if (node.isLeaf()) {
      double edgeDist = pointEdgeDistance(p, node.ball.edge);
      if (edgeDist < minDist) {
        return {edgeDist, node.ball.edge.id};
      }
      return {minDist, minId};
    }

    vector<pair<double, int>> childDistances;
    for (int i = 0; i < node.children.size(); i++) {
      double childDist = pointPointDistance(p, node.children[i].ball.centroid);
      childDistances.push_back({childDist, i});
    }

    sort(childDistances.begin(), childDistances.end());

    for (int i = 0; i < childDistances.size(); i++) {
      int childIndex = childDistances[i].second;
      pair<double, string> result = nearestEdgeRecursive(p, node.children[childIndex], minDist, minId);
      minDist = result.first;
      minId = result.second;
    }

    return {minDist, minId};
  }



  //! Guardar estructura dentro de un CSV
  void saveNodeToCSV(Node node, ofstream& file, int level) {
    if (node.isLeaf()) {
      file << level << "," << node.ball.centroid.x << "," << node.ball.centroid.y << "," << node.ball.radius << ","
      << node.ball.edge.a.x << "," << node.ball.edge.a.y << "," << node.ball.edge.b.x << "," << node.ball.edge.b.y << "\n";
    }
    else {
      file << level << "," << node.ball.centroid.x << "," << node.ball.centroid.y << "," << node.ball.radius << ",,,,\n";
    }
    for (const Node& child : node.children) {
      saveNodeToCSV(child, file, level + 1);
    }
  }
};


pair<double, double> extractCoords(string order) {
  int commaPos = order.find(',');

  if (commaPos == string::npos) {
    throw invalid_argument("Order invalida");
  }

  double firstNumber = stod(order.substr(0, commaPos));
  double secondNumber = stod(order.substr(commaPos + 1));

  return make_pair(firstNumber, secondNumber);
}

int main() {

  BallStarTree bst("points.csv", "edges.csv");
  //bst.saveToCSV("structure.csv");

  string order;
  cout << "Input the coords to search as: \"x,y\"" << endl;
  cout << "Input \"exit\" to stop" << endl << endl;
  while (cin >> order) {
    if (order == "exit") {
      break;
    }
    else {
      pair<double, double> coords = extractCoords(order);
      cout << "Closest edge with ID: " << bst.nearestEdge(Point(coords.first,coords.second)) << endl;
    }
    cout << endl;

    cout << "Input the coords to search as: \"x,y\"." << endl;
    cout << "Input \"exit\" to stop." << endl << endl;
  }
}

/*
-16.4168968,-71.5333161
-16.4168407,-71.5331871
-16.4024361,-71.5243656
-16.3767476,-71.5318152
-16.3857001,-71.4791703
*/

/*
-16.4126564,-71.5469335
*/