int rows = 99;
int columns = 99;

int posX = 1;
int posY = 0;

boolean[][] cells;

boolean[][] visited;

float cellWidth, cellHeight;

boolean keyClicked = false;
int keyTimer;

void setup() {
  
  size(1000, 1000);
  
  cellWidth = width / rows;
  cellHeight = height / rows;
  
  cells = new boolean[rows][columns];
  
  visited = new boolean[rows][columns];
  
  cells[1][0] = true;
  cells[rows - 2][columns - 1] = true;
  
  for (int x = 1; x < rows; x += 2) {
    for (int y = 1; y < columns; y += 2) {
      
      cells[x][y] = true;
    }
  }
  
  generateMaze();
  
  //render
  background(7, 54, 66);
  
  for (int x = 0; x < rows; x++) {
    for (int y = 0; y < columns; y++) {
      
      noStroke();
      
      if (cells[x][y]) { 
        fill(101, 123, 131);
        rect(x * cellWidth, y * cellHeight, cellWidth, cellHeight);
      }
    }
  }
}

PVector hunt() {
  
  PVector p = new PVector(0, 0);
  
  for (int x = 1; x < rows; x += 2) {
    for (int y = 1; y < columns; y += 2) {
      
      if (adjacentWalls(x, y) == 4) {
        p = new PVector(x, y);
        return p;
      }
    }
  }
  
  return p;
}

void kill(int x, int y) {

  if (adjacentWalls(x, y) <= 1)
    return;
  
  int i = floor(random(1, 5));
    
    //north
  if (i == 1) {
    
    if (y - 1 == 0) {
      return;
        
    } else {
      cells[x][y - 1] = true;
      kill(x, y - 2);
    }
     
    //east
  } else if (i == 2) {
    
    if (x + 1 == cells.length - 1) {
      return;
      
    } else {
      
      cells[x + 1][y] = true;
      kill(x + 2, y);
    }
      
    //south
  } else if (i == 3) {
    
    if (y + 1 == cells[0].length - 1) {
      return;
      
    } else {
      
      cells[x][y + 1] = true;
      kill(x, y + 2);
    }
      
    //west
  } else if (i == 4) {
      
    if (x - 1 == 0) {
      return;
      
    } else { 
      
      cells[x - 1][y] = true;
      kill(x - 2, y);
    }
  }
  
  return;
}

void correctPath(int x, int y, int prevDirection) {
  
  if (x == rows - 2 && y == columns - 2)
    return;
    
  int i = floor(random(1, 7));
  
  if (i <= 1) {
    
    if (prevDirection == 3)
      correctPath(x, y, 3);
      
    else if (y - 1 == 0) {
      correctPath(x, y, prevDirection);
        
    } else {
      cells[x][y - 1] = true;
      correctPath(x, y - 2, 1);
    }
    
  } else if (i <= 3) {
    
    if (prevDirection == 4)
      correctPath(x, y, 4);
      
    else if (x + 1 == cells.length - 1) {
      correctPath(x, y, prevDirection);
      
    } else {
      
      cells[x + 1][y] = true;
      correctPath(x + 2, y, 2);
    }
    
  } else if (i <= 5) {
    
    if (prevDirection == 1)
      correctPath(x, y, 1);
      
    else if (y + 1 == cells[0].length - 1) {
      correctPath(x, y, prevDirection);
      
    } else {
      
      cells[x][y + 1] = true;
      correctPath(x, y + 2, 3);
    }
    
  } else if (i <= 6) {
    
    if (prevDirection == 2)
      correctPath(x, y, 2);
      
    else if (x - 1 == 0) {
      correctPath(x, y, prevDirection);
      
    } else { 
      
      cells[x - 1][y] = true;
      correctPath(x - 2, y, 4);
    }
  }
  
  return;
}

int adjacentWalls(int x, int y) {
  
  int adj = 0;
  
  for (int i = -1; i <= 1; i++) {
    for (int j = -1; j <= 1; j++) {
      
      if (i == 0 || j == 0) {
        if (!cells[x + i][y + j])
          adj++;
      }
    }
  }
  
  return adj;
}

void generateMaze() {
  
  correctPath(1, 1, 3);
 
  while (!(hunt().x == 0 && hunt().y == 0)) {

      kill((int)hunt().x, (int)hunt().y);
    
  }
  
  cells[rows - 3][columns - 2] = true;
}

void draw() {
  
  noStroke();
  
  if (keyPressed && !keyClicked) {
    
    fill(42, 161, 152);
    rect(posX * cellWidth, posY * cellHeight, cellWidth, cellHeight);
    
    visited[posX][posY] = true;
    
    if (keyCode == DOWN || key == 's')
      if (cells[posX][posY + 1]) {
        posY += 1;
        keyClicked = true;
      }
    
    if (keyCode == UP || key == 'w')
      if (cells[posX][posY - 1]) {
        posY -= 1;
        keyClicked = true;
      }
    
    if (keyCode == LEFT || key == 'a')
      if (cells[posX - 1][posY]) {
        posX -= 1;
        keyClicked = true;
      }
    
    if (keyCode == RIGHT || key == 'd')
      if (cells[posX + 1][posY]) {
        posX += 1;
        keyClicked = true;
      }
  }
  
  if (keyClicked) {
    keyTimer++;
    
    if (keyTimer > 6) {
      keyTimer = 0;
      keyClicked = false;
    }
  }
  
  fill(203, 75, 22);
  rect(posX * cellWidth, posY * cellWidth, cellWidth, cellWidth);
}
