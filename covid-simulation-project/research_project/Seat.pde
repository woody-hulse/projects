class Seat {
  
  float x, y;
  boolean open;
  boolean containsInfected;
  
  Seat (float x, float y) {
    this.x = x;
    this.y = y;
    
    open = true;
    containsInfected = false;
  }
  
  void display() {
    if (open)
      containsInfected = false;
    
    float s = 3f / 55 * zoneLength;
    fill(120, 50);
    noStroke();
    rect(x*zoneLength-s, y*zoneLength-s, s*2, s*2);
  }
}
