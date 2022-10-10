class Button {
  
  PVector pos;
  float size;
  String symbol;
  
  boolean mouseOver;
  boolean pressed;
  
  Button(PVector pos, float size, String symbol) {
    this.pos = pos;
    this.size = size;
    this.symbol = symbol;
  }
  
  boolean mouseOver() {
    if (dist(mouseX, mouseY, pos.x, pos.y) < size) {
      return true;
    }
    return false;
  }
  
  void display() {
    noStroke();
    if (mouseOver()) {
      fill(foreground);
    } else {
      fill(130);
    }
    //ellipse(pos.x, pos.y, size, size);
    rect(pos.x - size/2, pos.y - size/2, size, size, size/4);
    
    fill(30);
    textAlign(CENTER, CENTER);
    textSize(size*0.4);
    text(symbol, pos.x, pos.y);
  }
}
