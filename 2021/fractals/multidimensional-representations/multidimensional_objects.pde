PVector[] vertex = new PVector[16384];
int vnum = 1;
int x = 0;
boolean overkill = false;

void setup() {
  size(2032, 1024);
  background(0);
  
  for (int i = 1; i <= 14; i++){
    for (int j = (int)pow(2, i-1); j < (int)pow(2, i); j++){
      vnum ++;
      
      if (j == 1){
        vertex[j] = new PVector(8, 8);
        vertex[j-1] = new PVector(8, 8);
      }
      
      if (i == 2){
        if (j == 2) vertex[j] = new PVector(8, 8);
        if (j == 3) vertex[j] = new PVector(8, 8);
      }
      
      if (i == 3) vertex[j] = new PVector(vertex[j%4].x+9, vertex[j%4].y+196);
      if (i == 4) vertex[j] = new PVector(vertex[j%8].x+16, vertex[j%8].y+169);
      if (i == 5) vertex[j] = new PVector(vertex[j%16].x+25, vertex[j%16].y+144);
      if (i == 6) vertex[j] = new PVector(vertex[j%32].x+36, vertex[j%32].y+121);
      if (i == 7) vertex[j] = new PVector(vertex[j%64].x+49, vertex[j%64].y+100);
      if (i == 8) vertex[j] = new PVector(vertex[j%128].x+64, vertex[j%128].y+81);
      if (i == 9) vertex[j] = new PVector(vertex[j%256].x+81, vertex[j%256].y+64);
      if (i == 10) vertex[j] = new PVector(vertex[j%512].x+100, vertex[j%512].y+49);
      if (i == 11) vertex[j] = new PVector(vertex[j%1024].x+121, vertex[j%1024].y+36);
      if (i == 12) vertex[j] = new PVector(vertex[j%2048].x+144, vertex[j%2048].y+25);
      if (i == 13) vertex[j] = new PVector(vertex[j%4096].x+169, vertex[j%4096].y+16);
      if (i == 14) vertex[j] = new PVector(vertex[j%8192].x+196, vertex[j%8192].y+9);
    }
  }
  
  for (int i = 0; i < vnum; i ++){
    stroke(255);
    strokeWeight(0.01);
    
    if(overkill == true){
      line(vertex[i].x, vertex[i].y, vertex[i].x, height-vertex[i].y);
      line(width-vertex[i].x, vertex[i].y, width-vertex[i].x, height-vertex[i].y);
      line(width-vertex[i].x, vertex[i].y, vertex[i].x, vertex[i].y);
    }
    
    if (i%4 == 0){
      line(vertex[i].x, vertex[i].y, vertex[i+1].x, vertex[i+1].y);
      line(vertex[i].x, vertex[i].y, vertex[i+2].x, vertex[i+2].y);
      line(vertex[i+3].x, vertex[i+3].y, vertex[i+1].x, vertex[i+1].y);
      line(vertex[i+3].x, vertex[i+3].y, vertex[i+2].x, vertex[i+2].y);
    }
    
    for (int j = 2; j <= 13; j++) if (vnum > i+pow(2, j) && i%pow(2, j+1) < pow(2, j)) line(vertex[i].x, vertex[i].y, vertex[i+(int)pow(2, j)].x, vertex[i+(int)pow(2, j)].y);
    
    if (overkill == true){
      for (int j = 2; j <= 13; j++) if (vnum > i+pow(2, j) && i%pow(2, j+1) < pow(2, j)) line(vertex[i].x, height-vertex[i].y, vertex[i+(int)pow(2, j)].x, height-vertex[i+(int)pow(2, j)].y);
      for (int j = 2; j <= 13; j++) if (vnum > i+pow(2, j) && i%pow(2, j+1) < pow(2, j)) line(width-vertex[i].x, vertex[i].y, width-vertex[i+(int)pow(2, j)].x, vertex[i+(int)pow(2, j)].y);
      for (int j = 2; j <= 13; j++) if (vnum > i+pow(2, j) && i%pow(2, j+1) < pow(2, j)) line(width-vertex[i].x, height-vertex[i].y, width-vertex[i+(int)pow(2, j)].x, height-vertex[i+(int)pow(2, j)].y);
    }

    noStroke();
    fill(255, 0, 0);
    ellipse(vertex[i].x, vertex[i].y, 2, 2);
    
    if(overkill == true){
      ellipse(vertex[i].x, height-vertex[i].y, 2, 2);
      ellipse(width-vertex[i].x, vertex[i].y, 2, 2);
      ellipse(width-vertex[i].x, height-vertex[i].y, 2, 2);
    }
  }
}
