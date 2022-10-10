class Classroom {
  int x, y;
  int l, h;
  
  String name;
  
  int numInCol;
  float d;
  
  int infectedCount;
  
  float percentageWindows;
  
  ArrayList<Seat> seats;
  
  Classroom (int x, int y, int l, int h, String name) {
    this.x = x;
    this.y = y;
    this.l = l;
    this.h = h;
    
    this.name = name;
    
    d = 0.5;
    numInCol = int(l/d) - 1;
    
    infectedCount = 0;
    
    seats = new ArrayList<Seat>();
    
    float d = 0.5;
    int numSeats = 0;
    for (float i = d; i < l; i+=d) {
      for (float j = d; j < h; j+=d) {
        if (numSeats < 25) {
          numSeats++;
          seats.add(new Seat(i*0.9+x, j+y));
        }
      }
    }
  }
  
  float getNumWindows() {
    // number of 0, 2, or 3 value walls around class
    int windows = 0;
    int total = 0;
    for (int i = x; i < x + l; i++) {
      total += 2;
      if (horizontalWalls[i-1][y] != 1)
        windows++;
      
      if (horizontalWalls[i][y+h] != 1)
        windows ++;
    }
    
    for (int j = y; j < y + h; j++) {
      total += 2;
      if (verticalWalls[x][j] != 1)
        windows++;
      
      if (verticalWalls[x+l][j] != 1)
        windows ++;
    }
    
    return float(windows) / total;
  }
  
  void getNumInfected() {
    int thoseInfected = 0;
    for (Seat s : seats) {
      if (s.containsInfected)
        thoseInfected ++;
    }
    infectedCount = thoseInfected;
  }
  
  void calculateNumWindows() {
    percentageWindows = getNumWindows();
  }
  
  void display() {
    
    for (Seat s : seats) {
      s.display();
    }
    
    noStroke();
    fill(200, 5);
    rect(x*zoneLength, y*zoneLength, l*zoneLength, h*zoneLength);
  }
}
