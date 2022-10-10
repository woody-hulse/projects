// before start, editing can occur and simulation does not begin
boolean start = true;
// editing none, walls, classrooms, or paths
int editing = 0;
// for wall selection
  int selectedVWallX = -1;
  int selectedVWallY = -1;
  int selectedHWallX = -1;
  int selectedHWallY = -1;
  
// previous selection
  int psvx = -1;
  int psvy = -1;
  int pshx = -1;
  int pshy = -1;
  
// list of buttons
ArrayList<Button> buttons;

// is the mouse currently being clicked
boolean isMouseClicked = false;
// is user drawing a new class
boolean drawingClass = false;
  int classX = -1;
  int classY = -1;
  int classL = 0;
  int classH = 0;
// is user drawing a path
boolean drawingPath = false;
  int[][][] newRoute = {{}, {}, {}, {}};
  int newPathIndex = 0;

// each of the 100 squares that divide the screeen
ArrayList<Boid>[][] zones;
// vertical wall boolean array
int[][] verticalWalls;
// horizontal wall boolean array
int[][] horizontalWalls;
// how long the side of each zone is
int zoneLength = 55;
//how long spacing is on bootom half of the screen
float spacingLength = 55;
// number of immune boids
int numImmune = 0;
// how fast the simulation is
int ticks = 0;
// length of each tick in seconds
float tickLength = 0.05 / 3;
// speed of ticks (mutable)
float tickSpeed = 2;
// factor of pixels to feet
float scale = 10.0 / zoneLength * 0.305;
// amount of time spent in class
int classTime = 0;
// number of boids seated
int numSeated = 0;

float totalSeconds = 0;

// list of infection probabilities over time
FloatList probabilities;
// scale of probability graph (starts at 0.001)
float pScale = 0.001;

PFont font1;

// previous second (for boid spawning)
int psecond = 0;

// is hall pattern circular
boolean circularHallPattern = true;

// determines if the three graphs are made
boolean detailedGraphs = true;

// averge number of contacts
float avgNumContacts = 0;
  FloatList anc_list;
  float panc = 0;
  float highest_anc = 0;
// total time of interacitons between boids
float totalInteractionTime = 0;
  FloatList it_list;
  float pit = 0;
  float highest_it = 0;
// total time of interactions with infected boids
float totalInfectedInteractionTime = 0;
  FloatList iit_list;
  float piit = 0;
  float highest_iit = 0;
// total time of interactions adjusted for distance
float totalWeightedInteractionTime = 0;
  FloatList wit_list;
  float pwit = 0;
  float highest_wit = 0;
// total weighted time of interactions with infected boid
float totalWeightedInfectedInteractionTime = 0;
  FloatList wiit_list;
  float pwiit = 0;
  float highest_wiit = 0;

// number of people wearing masks
int numWearingMasks = 0;
float boidAvoidanceFactor = 0;
float boidAvoidanceRange = 0;

// stats of selected boid
StringList selectedStats;
Boid selected;

// total transmission probability of the system over time t (mean likely number of individuals infected)
float transmissionProbability = 0;
// each statistic that is tracked over time in SEARID model
// int numSusceptible, numExposed, numRecovered, numInfected, numDead, numAsymptomatic;

// wall layout of building
// vertical|horizontal for each cell
// 28 columns
int[][] vWallsOfBuilding = {{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                            {0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1},
                            {0, 1, 0, 0, 1, 0, 0, 2, 0, 2, 0, 0, 1, 1, 0, 0, 2, 2, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1},
                            {0, 1, 0, 0, 1, 0, 0, 3, 1, 3, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1},
                            {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1},
                            {0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1},
                            {0, 0, 0, 0, 1, 1, 0, 1, 0, 2, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 2, 0, 0, 1},
                            {0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1}};
                            
int[][] hWallsOfBuilding = {{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                            {0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0},
                            {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                            {0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                            {0, 2, 1, 1, 1, 1, 3, 1, 1, 2, 1, 1, 0, 1, 1, 3, 1, 2, 1, 2, 1, 1, 1, 0, 0, 0, 0, 0},
                            {0, 0, 0, 0, 2, 1, 3, 1, 3, 2, 1, 1, 1, 1, 3, 1, 1, 3, 1, 2, 1, 1, 0, 0, 0, 0, 0, 0},
                            {0, 0, 0, 0, 2, 1, 3, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 2, 1, 0, 0, 0, 1, 0, 0, 0, 0},
                            {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 2, 1, 0, 0, 0, 1, 3, 1, 1, 0},
                            {0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0}};

// classroom to classroom paths
// junior paths
int[][] e_A82 = {{0,4},{19,4},{19,3}};
int[][] A82_A67 = {{19,3},{19,4},{14,4},{14,5}};
int[][] A67_A76 = {{14,5},{14,4},{17,4},{17,3}};
int[][] A76_e = {{17,3},{17,4},{0,4}};
int[][] A67_A69 = {{9,5},{9,3}};
int[][] A69_e = {{9,3},{9,4},{0,4}};
int[][] A67_A70 = {{14,5},{14,4},{15,4},{15,3}};
int[][] A70_e = {{15,3},{15,4},{0,4}};
int[][] e_A73 = {{0,4},{17,4},{17,5}};
int[][] A73_A67 = {{17,5},{17,4},{14,4},{14,5}};
int[][] e_A67 = {{0,4},{14,4},{14,5}};
int[][] A82_A70 = {{19,3},{19,4},{15,4},{15,3}};
int[][] A82_A69 = {{19,3},{19,4},{9,4},{9,3}};
int[][] A73_A70 ={{17,5},{17,4},{15,4},{15,3}};

// senior paths
int[][] A82_A30 = {{19,3},{19,4},{1,4},{1,3}};
int[][] A30_A73 = {{1,3},{1,4},{17,4},{17,5}};
int[][] e_A30 = {{0,4},{1,4},{1,3}};
int[][] A30_A76 = {{1,3},{1,4},{17,4},{17,3}};
int[][] A76_A61 = {{17,3},{17,4},{6,4},{6,3}};
int[][] A30_C2 = {{1,3},{1,4},{22,4},{22,7},{24,7},{24,6}};
int[][] C2_e = {{24,6},{24,7},{22,7},{22,4},{0,4}};
int[][] A30_A61 = {{1,3},{1,4},{6,4},{6,3}};
int[][] A61_e = {{6,3},{6,4},{0,4}};
int[][] A76_C2 = {{17,3},{17,4},{22,4},{22,7},{24,7},{24,6}};
int[][] A76_A73 = {{17,3},{17,5}};
int[][] A73_e = {{17,5},{17,4},{0,4}};

ArrayList<Classroom> classrooms;
int classroomNamingIndex = 1;

// textboxes (for renaming classrooms)
ArrayList<Textbox> textboxes;
                            
int block = 0;
int numBlocks = 3;

int[] routeNums = {
                      3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,// juniors
                      5, 5, 5, 5, 5, 5//seniors
                  };
int[][][][] routes = {
                        {e_A82, A82_A67, A67_A76, A76_e},
                        {e_A82, A82_A67, A67_A69, A69_e},
                        {reversePath(A76_e), reversePath(A67_A76), A67_A76, A76_e},
                        {reversePath(A76_e), reversePath(A67_A76), A67_A69, A69_e},
                        {reversePath(A76_e), reversePath(A67_A76), A67_A70, A70_e},
                        {e_A73, A73_A67, A67_A76, A76_e},
                        {e_A73, A73_A67, A67_A70, A70_e},
                        {e_A73, A73_A67, A67_A69, A69_e},
                        {e_A67, reversePath(A82_A67), A82_A70, A70_e},
                        {e_A67, reversePath(A73_A67), A73_A70, A70_e},
                        {e_A67, reversePath(A82_A67), A82_A69, A69_e},
                        
                        {e_A82, A82_A30, A30_A73, reversePath(e_A73)},
                        {e_A82, A82_A30, A30_C2, C2_e},
                        {e_A82, A82_A30, A30_A61, A61_e},
                        {e_A30, A30_A76, A76_A61, A61_e},
                        {e_A30, A30_A76, A76_C2, C2_e},
                        {e_A30, A30_A76, A76_A73, A73_e},
                     };
           
// foreground and background grey colors
color foreground = color(160);
color background = color(60);

// ability to pause simulation
boolean pause = false;

// reverses a path
int[][] reversePath(int[][] path) {
  int[][] newPath = new int[path.length][path[0].length];
  
  for (int i = 0; i < path.length; i++)
    newPath[i] = path[path.length-i-1];
   
  return newPath;
}

int[] addElement(int[] r, int val) {
  int[] arr = new int[r.length+1];
  
  for (int i = 0; i < r.length; i++) {
    arr[i] = r[i];
  }
  
  arr[arr.length-1] = val;
  
  return arr;
}

int[] removeElement(int[] r, int index) {
  int[] arr = new int[r.length-1];
  
  int addIndex = 0;
  for (int i = 0; i < r.length; i++) {
    if (i != index) {
      arr[addIndex] = r[i];
      addIndex++;
    }
  }
  
  return arr;
}

int[][][] removeWayPoint(int[][][] r) {
  int[][][] arr = new int[r.length][][];
  
  int[][] newPath = new int[r[newPathIndex].length - 1][2];
  for (int i = 0; i < r[newPathIndex].length - 1; i++) {
    newPath[i] = r[newPathIndex][i];
  }
  
  for (int i = 0; i < r.length; i++) {
    if (i == newPathIndex) {
      arr[i] = newPath;
    } else {
      arr[i] = r[i];
    }
  }
  
  return arr;
}

int[][][] addWayPoint(int[][][] r, int x, int y) {
  int[][][] arr = new int[r.length][][];
  int[] newElement = {x, y};
  
  int[][] lastPath = new int[r[newPathIndex].length + 1][2];
  for (int i = 0; i < lastPath.length; i++) {
    if (i < lastPath.length-1) {
      lastPath[i] = r[newPathIndex][i];
    } else {
      lastPath[i] = newElement;
    }
  }
  
  for (int i = 0; i < r.length; i++) {
    if (i == newPathIndex) {
      arr[i] = lastPath;
    } else {
      arr[i] = r[i];
    }
  }
  //arr[arr.length-1][arr[arr.length-1].length-1] = newElement;
  
  return arr;
}

int[][][][] addRoute(int[][][][] r, int[][][] newR) {
  int[][][][] newRoutes = new int[r.length+1][][][];

  for (int i = 0; i < r.length; i++) {
    newRoutes[i] = r[i];
  }
  
  newRoutes[newRoutes.length-1] = newR;
  
  return newRoutes;
}

int[][][][] removeRoute(int[][][][] r, int index) {
  int[][][][] newRoutes = new int[r.length-1][][][];
  
  int addIndex = 0;
  for (int i = 0; i < r.length; i++) {
    if (i != index) {
      newRoutes[addIndex] = r[i];
      addIndex++;
    }
  }
  
  return newRoutes;
}

// gets the sum of the elements in an array
int arraySum(int[] arr) {
  int sum = 0;
  for (int i = 0; i < arr.length; i++) {
    sum += arr[i];
  }
  return sum;
}

// checks if an item is in an aray
boolean inList(IntList list, int val) {
  for (int item : list) {
    if (item == val)
      return true;
  }
  return false;
}

// define total number of boids
int numBoids = arraySum(routeNums);

// spawns boids
int boidIndex = 0;
int[] enteredNums = new int[routeNums.length];

// characteristics
int numInfected = 1;
int percentImmune = 0;
int percentWearingMasks = 100;
float socialDistancingRange = 2;
float socialDistancingFactor = 0.2;

float ap = 0;
void spawnBoid() {
  int routeIndex = -1;
  while (routeIndex == -1) {
    int i = floor(random(0, routeNums.length));
    if (routeNums[i] > enteredNums[i]) {
      routeIndex = i;
      enteredNums[i]++;
    }
  }
  
  // temp to add random infected boid
  String condition = "Susceptible";
  if (arraySum(enteredNums) % floor(numBoids/numInfected) == 0) {
    condition = "Infected";
  }
  
  int sex = floor(random(0, 2));
  
  String year;
  if (routeIndex > 10) year = "Senior";
  else year = "Junior";
  
  // immunity
  if (condition == "Susceptible" && random(0, 1) < percentImmune/100f) {
    condition = "Immune";
    numImmune ++;
  }
  
  IntList empty = new IntList();
  Boid b = new Boid(
                    new PVector(0.5*zoneLength, 4.5*zoneLength),
                    new PVector(1, 0),
                    condition,
                    "Student",
                    (random(0, 100) < percentWearingMasks),
                    routes[routeIndex],
                    routes[routeIndex][0],
                    0,
                    false,
                    new PVector(0, 0),
                    boidIndex,
                    empty,
                    false,
                    sex,
                    year,
                    0
                   );
  boidIndex++;
  
  if (b.wearingMask) numWearingMasks++;
  boidAvoidanceFactor = b.avoidance;
  boidAvoidanceRange = b.avoidanceRange;
  
  zones[0][0].add(b);
}

void setup() {
  fullScreen();

  // initialize zones
  zones = new ArrayList[floor(width/zoneLength)][floor(height/2/zoneLength)];
  for (int i = 0; i < zones.length; i++) {
    for (int j = 0; j < zones[i].length; j++) {
      zones[i][j] = new ArrayList<Boid>();
    }
  }
  
  buttons = new ArrayList<Button>();
  // go
  buttons.add(new Button(new PVector(width - spacingLength, height - spacingLength), spacingLength, "►"));
  
  // pixel size
  buttons.add(new Button(new PVector(width/4 - spacingLength/4, height/2 + spacingLength*0.5), spacingLength/3, "►"));
  buttons.add(new Button(new PVector(width/4 - spacingLength*2, height/2 + spacingLength*0.5), spacingLength/3, "◄"));
  
  // editing
  buttons.add(new Button(new PVector(width/4 - spacingLength/4, height/2 + spacingLength*1.5), spacingLength/3, "►"));
  buttons.add(new Button(new PVector(width/4 - spacingLength*2.5, height/2 + spacingLength*1.5), spacingLength/3, "◄"));
  
  // wearing masks, immunity, social distancing factor, social distancing range
  for (int i = 0; i < 5; i++) {
    buttons.add(new Button(new PVector(width/4 - spacingLength/4, height/2 + spacingLength*(2.5 + i)), spacingLength/3, "►"));
    buttons.add(new Button(new PVector(width/4 - spacingLength*2, height/2 + spacingLength*(2.5 + i)), spacingLength/3, "◄"));
  }
  
  // initialize wall arrays
  verticalWalls = new int[zones.length][zones[0].length];
  horizontalWalls = new int[zones.length][zones[0].length];
  
  // initialize floatlists
  probabilities = new FloatList();
  anc_list = new FloatList();
  it_list = new FloatList();
  iit_list = new FloatList();
  wit_list = new FloatList();
  wiit_list = new FloatList();
  
  selectedStats = new StringList();
  
  // create classrooms
  classrooms = new ArrayList<Classroom>();
  // classroom locations and dimensions
  int[][] class_loc = {{1,1},{4,1},{9,1},{13,1},{17,1},{19,1},{23,1},{7,5},{9,5},{16,5},{20,5}};
  int[][] class_dim = {{3,3},{3,3},{3,3},{3,3},{2,3},{3,3},{4,5},{2,3},{6,3},{2,3},{2,3}};
  for (int i = 0; i < class_loc.length; i++) {
    classrooms.add(new Classroom(class_loc[i][0], class_loc[i][1], class_dim[i][0], class_dim[i][1], "Classroom " + classroomNamingIndex));
    classroomNamingIndex ++;
  }
  
  textboxes = new ArrayList<Textbox>();

  // draw walls
  for (int i = 0; i < zones.length; i ++) {
    for (int j = 0; j < zones[0].length; j ++) {
      if (j < vWallsOfBuilding.length && i < vWallsOfBuilding[0].length) {
        verticalWalls[i][j] = vWallsOfBuilding[j][i];
      }
      if (j < hWallsOfBuilding.length && i < hWallsOfBuilding[0].length) {
        horizontalWalls[i][j] = hWallsOfBuilding[j][i];
      }
    }
  }
}

StringList getStats(Boid b) {
  StringList s = new StringList();
  
  s.append(str(b.index));
  s.append(b.type);
  s.append(str(b.sex));
  s.append(b.year);
  s.append(b.condition);
  s.append(booleanText(b.wearingMask));
  s.append(str(b.velocity.mag()));
  s.append(str(b.path.length));
  s.append(str(b.interactions.size()));
  s.append(str(b.route[0][b.route[0].length-1][0]) + " " +
           str(b.route[0][b.route[0].length-1][1]) + " " +
           str(b.route[1][b.route[1].length-1][0]) + " " +
           str(b.route[1][b.route[1].length-1][1]) + " " +
           str(b.route[2][b.route[2].length-1][0]) + " " +
           str(b.route[2][b.route[2].length-1][1]) + " " +
           str(b.route[3][b.route[3].length-1][0]) + " " +
           str(b.route[3][b.route[3].length-1][1]));
  
  return s;
}

String[] classroomsInRoute(int[][][] route) {
  String[] classes = new String[numBlocks];
  
  return classes;
}

// deselects all other boids
void deselectAll() {
  for (int i = 0; i < zones.length; i++) {
    for (int j = 0; j < zones[i].length; j++) {  
      for (Boid b : zones[i][j])
        b.selected = false;
    }
  }
}

void addTeachers() {
  // add teachers
  for (int i = 0; i < classrooms.size(); i++) {
    Classroom c = classrooms.get(i);
    IntList empty2 = new IntList();
    int sex = floor(random(0, 2));
    Boid b = new Boid(
                      new PVector((c.x + c.l - 0.15)*zoneLength, (c.y + c.h/2 + 0.5)*zoneLength),
                      new PVector(1, 0),
                      "Immune",
                      "Teacher",
                      true,
                      routes[0],
                      routes[0][0],
                      0,
                      true,
                      new PVector((c.x + c.l - 0.15)*zoneLength, (c.y + c.h/2 + 0.5)*zoneLength),
                      i,
                      empty2,
                      false,
                      sex,
                      "",
                      0
                      );
     zones[0][0].add(b);
  }
}

void keyReleased() {
  if (!start) {
    if (pause) {
      pause = false;
      deselectAll();
    } else pause = true;
  }
}

// for drawing paths
void mouseReleased() {
  if (editing == 3) {
    if (drawingPath && mouseY < height/2) {
      newPathIndex++;
    }
  }
}

// select a boid
void mouseClicked() {
  
  isMouseClicked = true;
 
  int zoneX = floor(mouseX / zoneLength);
  int zoneY = floor(mouseY / zoneLength);
  
  if (start) {
    
    for (int i = 0; i < buttons.size(); i++) {
      Button b = buttons.get(i);
      
      // button programming
      if (b.mouseOver()) {
        if (i == 0) {
          start = false;
          pause = false;
          reset();
          addTeachers();
          scale = 10.0 / zoneLength * 0.305;
          for (Classroom c : classrooms) {
            c.calculateNumWindows();
          }
        } else if (i == 2 && zoneLength > 10) {
          zoneLength --;
          
          verticalWalls = resize2DArray(verticalWalls);
          horizontalWalls = resize2DArray(horizontalWalls);
          
          // re-initialize zones
          zones = new ArrayList[floor(width/zoneLength)][floor((height/2-spacingLength/2)/zoneLength)];
          for (int m = 0; m < zones.length; m++) {
            for (int k = 0; k < zones[i].length; k++) {
              zones[m][k] = new ArrayList<Boid>();
            }
          }
          
        } else if (i == 1 && zoneLength < 99) {
          zoneLength ++;
          
          verticalWalls = resize2DArray(verticalWalls);
          horizontalWalls = resize2DArray(horizontalWalls);
          
          // re-initialize zones
          zones = new ArrayList[floor(width/zoneLength)][floor((height/2-spacingLength/2)/zoneLength)];
          for (int m = 0; m < zones.length; m++) {
            for (int k = 0; k < zones[i].length; k++) {
              zones[m][k] = new ArrayList<Boid>();
            }
          }
        } else if (i == 3) {
          editing = (editing + 1) % 4;
        } else if (i == 4) {
          editing = (editing + 3) % 4;
        } else if (i == 5 && numInfected < numBoids) numInfected++;
        else if (i == 6 && numInfected > 0) numInfected --;
        else if (i == 7 && percentWearingMasks < 100) percentWearingMasks ++;
        else if (i == 8 && percentWearingMasks > 0) percentWearingMasks--;
        else if (i == 9 && percentImmune < 100) percentImmune ++;
        else if (i == 10 && percentImmune > 0) percentImmune--;
        else if (i == 11 && socialDistancingFactor < 4) socialDistancingFactor += 0.05;
        else if (i == 12 && socialDistancingFactor > 0.002) socialDistancingFactor -= 0.05;
        else if (i == 13 && socialDistancingRange < 5) socialDistancingRange += 0.5;
        else if (i == 14 && socialDistancingRange > 0) socialDistancingRange -= 0.5;
      }
    }
    
    
  } else {
    deselectAll();
    selectedStats = new StringList();
    
    if (zoneX < zones.length && zoneY < zones[0].length) {
      for (Boid b : zones[zoneX][zoneY]) {
        if (dist(mouseX, mouseY, b.position.x, b.position.y) < b.size*1.5)
        {
          b.selected = true;
          selected = b;
          selectedStats = getStats(b);
        }
      }
    }
  }
}

// resizes the cell arrays
int[][] resize2DArray(int[][] oldArray) {
  int newX = floor(width/zoneLength);
  int newY = floor(height/2/zoneLength);
  int[][] newArray = new int[newX][newY];
  
  for (int i = 0; i < newX; i++) {
    for (int j = 0; j < newY; j++) {
      if (i < oldArray.length && j < oldArray[0].length) {
        newArray[i][j] = oldArray[i][j];
      } else {
        newArray[i][j] = 0;
      }
    }
  }
  
  return newArray;
}

void drawGraph(FloatList list, PVector pos, PVector size, float highest, color c) {
  noStroke();
  fill(200, 10);
  if (c == color(180, 50, 50))
    rect(pos.x, pos.y, size.x, size.y);
  
  strokeWeight(1);
  stroke(c);
  if (list.size() > 2) {
    for (int i = 1; i < list.size(); i++) { 
      float p1 = list.get(i - 1);
      float p2 = list.get(i);
      float scl = highest*1.5;
      line(pos.x + size.x * i/list.size(), pos.y + (1 - p1/scl)*size.y, pos.x + size.x * (i + 1)/list.size(), pos.y + (1 - p2/scl)*size.y);
    }
  }
  noStroke();
}

// makes boolean text capitalized
String booleanText(boolean b) {
  if (b) return "True";
  else return "False";
}
// converts seconds to time
String convertTime(int s) {
  int hours = floor(s/3600);
  int minutes = floor(s/60) - hours*60;
  int seconds = s%60;
  
  String time = nf(hours, 1) + ":" + nf(minutes, 2) + ":" + nf(seconds, 2);
  
  return time;
}

// resets all necessary stats
void reset() {
  ticks = 0;
  totalSeconds = 0;
  tickSpeed = 2;
  block = 0;
  numWearingMasks = 0;
  transmissionProbability = 0;
  
  enteredNums = new int[routeNums.length];
  
  for (int i = 0; i < zones.length; i++) {
    for (int j = 0; j < zones[i].length; j++) {  
      zones[i][j].clear();
    }
  }
  
  for (Classroom c : classrooms) {
    for (Seat s : c.seats) {
      s.open = true;
    }
  }
  
  probabilities.clear();
  anc_list.clear();
  it_list.clear();
  iit_list.clear();
  wit_list.clear();
  wiit_list.clear();
  
  avgNumContacts = 0;
  panc = 0;
  highest_anc = 0;
  totalInteractionTime = 0;
  pit = 0;
  highest_it = 0;
  totalInfectedInteractionTime = 0;
  piit = 0;
  highest_iit = 0;
  totalWeightedInteractionTime = 0;
  pwit = 0;
  highest_wit = 0;
  totalWeightedInfectedInteractionTime = 0;
   pwiit = 0;
  highest_wiit = 0;
  numImmune = 0;
  
  numWearingMasks = 0;
  boidAvoidanceFactor = 0;
  boidAvoidanceRange = 0;
  
  numSeated = 0;
  
  
  selectedStats.clear();

}

void keyPressed() {
  if (start && editing == 2) {
    for (Textbox t : textboxes) {
      t.keyPress(key, (int)keyCode);
    }
  }
}

void draw() {
  
  background(30);
  
  if (start) {
    
    int zoneX = 0;
    int zoneY = 0;
    
    float distX = mouseX % zoneLength;
    float distY = mouseY % zoneLength;
    
    float dist = zoneLength/4;
    
    float highlightWidth = 4f / 55 * zoneLength;
    
    if (mouseY < height/2) {
      zoneX = floor(mouseX / zoneLength);
      zoneY = floor(mouseY / zoneLength);
    } else {
      zoneX = floor(mouseX / zoneLength);
      zoneY = floor((mouseY - height/2) / spacingLength);
    }
    
    // create stats page
    pushMatrix();
    translate(spacingLength, height/2);
    
    pushMatrix();
    float rowSize = 0;
    for (float i = 0; i < 8; i++) {
      translate(0, rowSize);
      rowSize = spacingLength;
      
      fill(30 + 10 * ((i+1)%2));
      rect(0, 0, width/4 - spacingLength/2, rowSize);
    }
    popMatrix();
    
    translate(spacingLength/8, -spacingLength/3);
    fill(foreground);
    textSize(15);
    textAlign(LEFT, CENTER);
    text("Pixel Size of Each Cell", 0, spacingLength*0.75);
    text("Editing", 0, spacingLength*1.75);
    text("Number of Infected Students", 0, spacingLength*2.75);
    text("% Wearing Masks", 0, spacingLength*3.75);
    text("% Immune", 0, spacingLength*4.75);
    text("Social Distancing Factor", 0, spacingLength*5.75);
    text("Social Distancing Range", 0, spacingLength*6.75);
    
    textAlign(CENTER, CENTER);
    textSize(15);
    text(zoneLength, width/4 - spacingLength*2.25, spacingLength*0.75);
    text(numInfected, width/4 - spacingLength*2.25, spacingLength*2.75);
    text(percentWearingMasks, width/4 - spacingLength*2.25, spacingLength*3.75);
    text(percentImmune, width/4 - spacingLength*2.25, spacingLength*4.75);
    text(nf(socialDistancingFactor, 0, 3), width/4 - spacingLength*2.25, spacingLength*5.75);
    text(nf(socialDistancingRange, 0, 1) + " m", width/4 - spacingLength*2.25, spacingLength*6.75);
    
    textSize(15);
    String editingText = "";
    if (editing == 0) editingText = "Nothing";
    else if (editing == 1) editingText = "Walls";
    else if (editing == 2) editingText = "Classrooms";
    else if (editing == 3) editingText = "Paths";
    
    text(editingText, width/4 - spacingLength*2.5, spacingLength*1.75);
    
    popMatrix();
    
    // display buttons
    for (Button b : buttons) {
      b.display();
    }
    
    if (editing == 1) {
      
      stroke(50);
      for (int i = 0; i < zones.length; i++)
        line(i*zoneLength, 0, i*zoneLength, zones[0].length*zoneLength);
      for (int i = 0; i < zones[0].length; i++)
        line(0, i*zoneLength, zones.length*zoneLength, i*zoneLength);
      
      if (mouseX < (verticalWalls.length-0.5)*zoneLength && mouseY < (horizontalWalls[0].length-0.5)*zoneLength) {
        
        fill(200, 50);
        noStroke();
        if (distY > dist && distY < zoneLength - dist) {
          if (distX < dist && zoneX < verticalWalls.length) {
            rect(zoneX * zoneLength - highlightWidth, zoneY * zoneLength, highlightWidth*2, zoneLength);
            selectedVWallX = zoneX;
            selectedVWallY = zoneY;
          } else if (distX > zoneLength - dist) {
            rect((zoneX+1) * zoneLength - highlightWidth, zoneY * zoneLength, highlightWidth*2, zoneLength);
            selectedVWallX = zoneX + 1;
            selectedVWallY = zoneY;
          } else {
            selectedVWallX = -1;
            selectedVWallY = -1;
          }
        } else {
          selectedVWallX = -1;
          selectedVWallY = -1;
        }
        
        if (distX > dist && distX < zoneLength - dist) {
          if (distY < dist && zoneY < horizontalWalls[0].length) {
            rect(zoneX * zoneLength, zoneY * zoneLength - highlightWidth, zoneLength, highlightWidth*2);
            selectedHWallX = zoneX;
            selectedHWallY = zoneY;
          } else if (distY > zoneLength - dist) {
            rect(zoneX * zoneLength, (zoneY+1) * zoneLength - highlightWidth, zoneLength, highlightWidth*2);
            selectedHWallX = zoneX;
            selectedHWallY = zoneY + 1;
          } else {
            selectedHWallX = -1;
            selectedHWallY = -1;
          }
        } else {
          selectedHWallX = -1;
          selectedHWallY = -1;
        }
      } else {
        selectedVWallX = -1;
        selectedVWallY = -1;
        selectedHWallX = -1;
        selectedHWallY = -1;
      }
      
      if (mousePressed) {
        if (mouseButton == LEFT) {
          if (selectedVWallX >= 0 && selectedVWallY >= 0 && (psvx != selectedVWallX || psvy != selectedVWallY)) {
            verticalWalls[selectedVWallX][selectedVWallY] = (verticalWalls[selectedVWallX][selectedVWallY] + 1) % 4;
            psvx = selectedVWallX;
            psvy = selectedVWallY;
          }
        
          if (selectedHWallX >= 0 && selectedHWallY >= 0 && (pshx != selectedHWallX || pshy != selectedHWallY)) {
            horizontalWalls[selectedHWallX][selectedHWallY] = (horizontalWalls[selectedHWallX][selectedHWallY] + 1) % 4;
            pshx = selectedHWallX;
            pshy = selectedHWallY;
          }
        } else if (mouseButton == RIGHT) {
          if (selectedVWallX >= 0 && selectedVWallY >= 0) verticalWalls[selectedVWallX][selectedVWallY] = 0;
          if (selectedHWallX >= 0 && selectedHWallY >= 0) horizontalWalls[selectedHWallX][selectedHWallY] = 0;
        }
      } else {
        psvx = -1;
        pshx = -1;
        psvy = -1;
        pshy = -1;
      }
    } else if (editing == 2) {
    
      // display classrooms
      for (int i = 0; i < classrooms.size(); i++) {
        Classroom c = classrooms.get(i);
        c.display();
        
        pushMatrix();
        float translationX = width/4 + spacingLength + width/4 * floor(i/8);
        float translationY = height/2 + (i%8) * spacingLength;
        translate(translationX, translationY);
        
        fill(30 + 10 * ((i+1)%2));
        rect(0, 0, width/4 - spacingLength/2, spacingLength);
        
        fill(foreground);
        textSize(20);
        textAlign(LEFT, CENTER);
        text(c.name, spacingLength/8, spacingLength*0.5);
        
        //textSize(13);
        //text("x = " + c.x + ", y = " + c.y + "\nl = " + c.l + ", h = " + c.h, spacingLength*3, spacingLength*0.5);
        
        // show classroom moused over
        if (mouseX > translationX && mouseY > translationY && mouseX < translationX + width/4 - spacingLength/2 && mouseY < translationY + spacingLength && !drawingClass) {
          fill(200, 10);
          rect(c.x*zoneLength - translationX, c.y*zoneLength - translationY, c.l*zoneLength, c.h*zoneLength);
          
          // rename classrooms
          if (mousePressed && mouseX < translationX + spacingLength * 6 && textboxes.size() == 0) {
            textboxes.add(new Textbox(new PVector(translationX, translationY), new PVector(spacingLength*6, spacingLength), i));
          }
        }
        
        if (dist(mouseX, mouseY, width/2 - spacingLength*0.3 + width/4 * floor(i/8), height/2 + (i%8) * spacingLength + spacingLength*0.5) < spacingLength*0.6 && textboxes.size() == 0) {
          fill(210, 55, 55);
          
          if (isMouseClicked) classrooms.remove(i);
          
        } else fill(180, 50, 50);
        rect(width/4 - spacingLength*1.3, spacingLength*0.2, spacingLength*0.6, spacingLength*0.6, 5);
        
        textAlign(CENTER, CENTER);
        fill(200);
        textSize(20);
        text("x", width/4 - spacingLength, spacingLength*0.4);
        
        popMatrix();
      }
      
      // remove if user has clicked away
      if (textboxes.size() == 1) {
        if (mousePressed && !textboxes.get(0).mouseOver())
          textboxes.remove(0);
      }
      
      for (int t = 0; t < textboxes.size(); t ++) {
        Textbox tb = textboxes.get(t);
        tb.display();
      }
      
      // new classroom button
      Button newClass = new Button(new PVector(width*3/8 + spacingLength + width/4 * floor(classrooms.size()/8), height/2 + (classrooms.size()%8 + 0.5) * spacingLength), spacingLength/2, "▼");
      
      if (isMouseClicked && newClass.mouseOver()) {
        drawingClass = true;
      } else if (!drawingClass) {
        newClass.display();
      }
      
      // draw a new class
      if (drawingClass) {
        if (mousePressed) {
          if (classX == -1 && mouseY < height/2) {
            classX = zoneX;
            classY = zoneY;
          }
          
          try {
            classL = zoneX - classX;
            classH = zoneY - classY;
            
            if (classL >= 0) classL ++;
            if (classH >= 0) classH ++;
          } catch (Exception ie) {
            classL = 0;
            classH = 0;
          }
          
          noStroke();
          fill(200, 50);
          rect(classX*zoneLength, classY*zoneLength, classL*zoneLength, classH*zoneLength);
        } else if (classX != -1) {
          if (classL < 0) {
            classX += classL;
            classL = abs(classL);
          }
          
          if (classH < 0) {
            classY += classH;
            classH = abs(classH);
          }
          
          drawingClass = false;
          classrooms.add(new Classroom(classX, classY, classL, classH, "Classroom " + classroomNamingIndex));
          classroomNamingIndex ++;
          classX = -1;
          classY = -1;
          classL = 0;
          classH = 0;
        }
      }
    } else if (editing == 3) {
      
      color b1 = color(180, 50, 50);
      color b2 = color(50, 180, 50);
      color b3 = color(50, 50, 180);
      color e = color(150);
      
      pushMatrix();
      translate(width/2, height/2 - spacingLength*0.8);
      fill(b1);
      rect(-spacingLength*6, 0, 15, 15, 5);
      fill(b2);
      rect(-spacingLength*3, 0, 15, 15, 5);
      fill(b3);
      rect(0, 0, 15, 15, 5);
      fill(e);
      rect(spacingLength*3, 0, 15, 15, 5);
      
      fill(foreground);
      textSize(15);
      textAlign(LEFT, CENTER);
      text("Block 1 Path", -spacingLength*6 + 30, 5);
      text("Block 2 Path", -spacingLength*3 + 30, 5);
      text("Block 3 Path", 30, 5);
      text("Exit Path", spacingLength*3 + 30, 5);
      popMatrix();
      
      for (int i = 0; i < routes.length; i++) {
        pushMatrix();
        float translationX = width/4 + spacingLength + width/4 * floor(i/16);
        float translationY = height/2 + (i%16) * spacingLength/2;
        translate(translationX, translationY);
        
        noStroke();
        fill(30 + 10 * ((i+1)%2));
        rect(0, 0, width/4 - spacingLength/2, spacingLength/2);
        
        fill(foreground);
        textSize(15);
        textAlign(LEFT, CENTER);
        text("Route " + (i+1), spacingLength/8, spacingLength*0.25);
        
        textAlign(CENTER, CENTER);
        text(routeNums[i], width/8, spacingLength*0.25);
        
        // change number of students in each route
        pushMatrix();
        translate(-translationX, -translationY);
        Button addRouteNum = new Button(new PVector(translationX + width/8 + spacingLength/2, translationY + spacingLength*0.25), spacingLength/4,  "►");
        Button subRouteNum = new Button(new PVector(translationX + width/8 - spacingLength/2, translationY + spacingLength*0.25), spacingLength/4,  "◄");
        
        addRouteNum.display();
        subRouteNum.display();
        
        if (isMouseClicked) {
          if (addRouteNum.mouseOver() && routeNums[i] < 99) {
            routeNums[i] ++;
            numBoids ++;
          }
          
          if (subRouteNum.mouseOver() && routeNums[i] > 0) {
            routeNums[i] --;
            numBoids --;
          }
        }
        popMatrix();
        
        fill(180, 50, 50);
        
        if (mouseX > translationX && mouseY > translationY && mouseX < translationX + width/4 - spacingLength/2 && mouseY < translationY + spacingLength/2 && !drawingPath) {
          for (int p = 0; p < routes[i].length; p++) {
            for (int w = 0; w < routes[i][p].length-1; w++) {
              strokeWeight(7 * zoneLength / 55);
              if (p == 0) stroke(b1);
              else if (p == 1) stroke(b2);
              else if (p == 2) stroke(b3);
              else if (p == 3) stroke(e);
              line(routes[i][p][w][0]*zoneLength - translationX + (spacingLength/2 + p*10 - 15) * zoneLength / 55, 
                   routes[i][p][w][1]*zoneLength - translationY + (spacingLength/2 + p*10 - 15) * zoneLength / 55, 
                   routes[i][p][w+1][0]*zoneLength - translationX + (spacingLength/2 + p*10 - 15) * zoneLength / 55, 
                   routes[i][p][w+1][1]*zoneLength - translationY + (spacingLength/2 + p*10 - 15) * zoneLength / 55);
            }
          }
          
          if (mouseX > translationX + width/4 - spacingLength) {
            
            if (isMouseClicked) {
              routes = removeRoute(routes, i);
              routeNums = removeElement(routeNums, i);
              numBoids = arraySum(routeNums);
            }
            
            fill(210, 55, 55);
          }
        }
        
        noStroke();
        rect(width/4 - spacingLength*0.9, spacingLength*0.1, spacingLength*0.3, spacingLength*0.3, 2);
        
        textAlign(CENTER, CENTER);
        fill(200);
        textSize(12);
        text("x", width/4 - spacingLength*0.75, spacingLength*0.2);
        
        popMatrix();
      }
      
      // new classroom button
      Button newPath = new Button(new PVector(width*3/8 + spacingLength + width/4 * floor(routes.length/16), height/2 + (routes.length%16 + 1) * spacingLength/2), spacingLength/2, "▼");
      
      if (isMouseClicked && newPath.mouseOver()) {
        drawingPath = true;
        newPathIndex = 0;
      } else if (!drawingPath) {
        newPath.display();
      }
      
      if (drawingPath) {
        if (mouseY < height/2) {
          
          //indicate where cursor is
          fill(200, 20);
          rect(zoneX*zoneLength, zoneY * zoneLength, zoneLength, zoneLength);
          
          if (mousePressed && newPathIndex < 4) {
            if (newRoute[newPathIndex].length == 0) {
              newRoute = addWayPoint(newRoute, zoneX, zoneY);
            } else {
              // check if mouse is in different zone
              if (zoneX != newRoute[newPathIndex][newRoute[newPathIndex].length-1][0] || 
                  zoneY != newRoute[newPathIndex][newRoute[newPathIndex].length-1][1]) {
                
                newRoute = addWayPoint(newRoute, zoneX, zoneY);
                 
                // check if loop was made
                if (newRoute[newPathIndex].length > 2) {
                  for (int i = 0; i < newRoute[newPathIndex].length - 1; i++) {
                     if (zoneX == newRoute[newPathIndex][i][0] && zoneY == newRoute[newPathIndex][i][1]) {
                       // delete loop
                       for (int j = 0; j < newRoute[newPathIndex].length - i; j++) {
                         newRoute = removeWayPoint(newRoute);
                       }
                     }
                  }
                }
              }
            }
          } else if (newPathIndex >= 4) {
            // add new route to existing ones
            routes = addRoute(routes, newRoute);
            routeNums = addElement(routeNums, 0);
            int[][][] blank = {{}, {}, {}, {}};
            newRoute = blank;
            drawingPath = false;
          }
        }
        
        // display new route
        for (int p = 0; p < newRoute.length; p++) {
          for (int w = 0; w < newRoute[p].length-1; w++) {
            strokeWeight(7 * zoneLength / 55);
            if (p == 0) stroke(b1);
            else if (p == 1) stroke(b2);
            else if (p == 2) stroke(b3);
            else if (p == 3) stroke(e);
            line(newRoute[p][w][0]*zoneLength + (spacingLength/2 + p*10 - 15) * zoneLength / 55, 
                 newRoute[p][w][1]*zoneLength + (spacingLength/2 + p*10 - 15) * zoneLength / 55, 
                 newRoute[p][w+1][0]*zoneLength + (spacingLength/2 + p*10 - 15) * zoneLength / 55, 
                 newRoute[p][w+1][1]*zoneLength + (spacingLength/2 + p*10 - 15) * zoneLength / 55);
          }
        }
      }
    }
  
  } else {
  
    if (block == numBlocks + 1)
      pause = true;
    
    // put in room numbers
    textSize(40);
    textAlign(CENTER, CENTER);
    fill(background);
    noStroke();
    
      // draw classrooms
    for (Classroom c : classrooms) {
      c.display();
    }
    
    if (!pause) {
      ticks++;
      totalSeconds += tickLength*tickSpeed;
    }
    int s = floor(totalSeconds);
    
    //   interval events
    // boids spawn at intervals
    int spawnInterval = 1;
    if (s <= numBoids*spawnInterval) {
      // check if second has changed
      if (s > psecond && s%spawnInterval == 0) {
        spawnBoid();
      }
    }
    
    
    
    // draw data
    textAlign(LEFT, CENTER);
    textSize(15);
    fill(foreground);
    noStroke();
    
    if (!pause) {
      if (detailedGraphs && s > psecond && tickSpeed != 200) {
        float anc_dif = avgNumContacts - panc;
        float it_dif = totalInteractionTime - pit;
        float iit_dif = totalInfectedInteractionTime - piit;
        float wit_dif = totalWeightedInteractionTime - pwit;
        
        if (it_dif < numBoids*2) {
          if (anc_dif >= 0) anc_list.append(anc_dif);
          it_list.append(it_dif);
          iit_list.append(iit_dif);
          wit_list.append(wit_dif);
        }
        
        panc = avgNumContacts;
        pit = totalInteractionTime;
        piit = totalInfectedInteractionTime;
        pwit = totalWeightedInteractionTime;
        
        if (anc_dif > highest_anc && it_dif < numBoids*2) highest_anc = anc_dif;
        if (it_dif > highest_it && it_dif < numBoids*2) highest_it = it_dif;
        if (iit_dif > highest_iit && it_dif < numBoids*2) highest_iit = iit_dif;
        if (wit_dif > highest_wit && it_dif < numBoids*2) highest_wit = wit_dif;
      }
      
      float wiit_dif = totalWeightedInfectedInteractionTime - pwiit;
      if (s > psecond && (tickSpeed != 200 || s%200 == 0))
        wiit_list.append(wiit_dif);
      pwiit = totalWeightedInfectedInteractionTime;
      if (wiit_dif > highest_wiit) highest_wiit = wiit_dif;
    }
    
    textAlign(CENTER);
    
    // accelerate time if everyone is seated
    if (numSeated == numBoids) {
      // checks number of infected boids in classrooms
      if (tickSpeed != 200) {
        for (Classroom c : classrooms) {
          c.getNumInfected();
        }
      }
      tickSpeed = 200;
    } else {
      tickSpeed = 2;
    }
    numSeated = 0;
    
    // change block over block intervals
    if (!pause) {
      int classTransition = 5*60;
      int[] classTimes = {45*60, 45*60, 90*60, 0};
      int compareTime = 0;
      for (int i = 0; i < block; i++) {
        compareTime += classTransition + classTimes[i];
      }
      if (s > psecond && s - classTransition - classTimes[block] - compareTime > 0){
        block ++;
        // reset all seats
        for (Classroom c : classrooms) {
          for (Seat seat : c.seats) {
            seat.open = true;
          }
        }
      }
    }
    
    // selection
    if (selectedStats.size() > 0) {
      selectedStats = getStats(selected);
      pushMatrix();
      translate(width/4*3, height/2);
      
      pushMatrix();
      for (int i = 0; i < 7; i ++) { 
        fill(30 + 10 * ((i+1)%2));
        rect(0, 0, width/4 - spacingLength/2, spacingLength/2);
        translate(0, spacingLength/2);
      }
      popMatrix();
      
      translate(spacingLength/8, -spacingLength/3);
      fill(foreground);
      textAlign(CENTER, CENTER);
      text(selectedStats.get(1) + " " + int(selectedStats.get(0)), width/8 - spacingLength/2, 0);
      textAlign(LEFT, CENTER);
      text("Sex", 0, spacingLength*0.5);
      text("Year", 0, spacingLength*1);
      text("Age", 0, spacingLength*1.5);
      text("Condition", 0, spacingLength*2);
      text("Wearing Mask", 0, spacingLength*2.5);
      text("State", 0, spacingLength*3);
      text("Number of Contacts", 0, spacingLength*3.5);
      
      textAlign(RIGHT, CENTER);
      translate(width/4 - spacingLength*3/4, 0);
      if (int(selectedStats.get(2)) == 1) text("M", 0, spacingLength*0.5);
      else text("F", 0, spacingLength*0.5);
      text(selectedStats.get(3), 0, spacingLength*1);
      if (selectedStats.get(3) == "Junior") text("17", 0, spacingLength*1.5);
      else if (selectedStats.get(3) == "Senior") text("18", 0, spacingLength*1.5);
      text(selectedStats.get(4), 0, spacingLength*2);
      text(selectedStats.get(5), 0, spacingLength*2.5);
      if (int(selectedStats.get(7)) > 0) text("Going to Class", 0, spacingLength*3);
      else if (float(selectedStats.get(6)) > 0.1) text("Finding Seat", 0, spacingLength*3);
      else text("In Class", 0, spacingLength*3);
      text(selectedStats.get(8), 0, spacingLength*3.5);
      
      popMatrix();
      
      // draw block circles
      color[] colors = {color(180, 50, 50),
                        color(50, 180, 50),
                        color(50, 50, 180),
                        color(150)
                       };
     
      String[] vals = split(selectedStats.get(9), " ");
      for (int i = 0; i < 8; i+=2) {
        fill(colors[i/2]);
        noStroke();
        ellipse(float(vals[i] + ".5")*zoneLength, float(vals[i+1] + ".5")*zoneLength, zoneLength/3, zoneLength/3);
      }
      
      textAlign(CENTER, DOWN);
    }
    
    
    // graph infection probability 
    // dimensions
    float pGraphX = spacingLength;
    float pGraphY = height/2;
    float pGraphSizeY = height/2 - spacingLength;
    float pGraphSizeX = width/2 - 2*spacingLength;
    
    // draw d/dx graph
    drawGraph(wiit_list, new PVector(pGraphX, pGraphY), new PVector(pGraphSizeX, pGraphSizeY), highest_wiit, color(100, 25, 25));
    
    stroke(foreground);
    fill(foreground);
    textSize(15);
    //graph title
    text("% Chance of Transmission vs. Time", pGraphX + pGraphSizeX/2, pGraphY - spacingLength/3);
    textSize(10);
    for (float i = 0; i <= 1.01; i+=.1)
    {
      text(1.5*transmissionProbability*i * 100, pGraphX/2, pGraphY + (1 - i)*pGraphSizeY);
    }
    //text(convertTime(int(s)), pGraphX + pGraphSizeX, pGraphY + pGraphSizeY + spacingLength/3);
    
    // add probability to list
    if (!pause && s > psecond && (tickSpeed != 200 || s%200 < 5))
        probabilities.append(transmissionProbability);
      
    // graph probabilties
    drawGraph(probabilities, new PVector(pGraphX, pGraphY), new PVector(pGraphSizeX, pGraphSizeY), transmissionProbability, color(180, 50, 50));
    
    if (s > 11760 || pause) {
      Button next = new Button(new PVector(width - spacingLength, height - spacingLength), spacingLength, "►");
      next.display();
      
      if (isMouseClicked && next.mouseOver()) {
        start = true;
      }
    }
    
    // set new psecond
    if (!pause)
      psecond = s;
  }
  
  if (!start && block < 3) avgNumContacts = 0;
    
  // loop through to display each of the zones (light up to indicate things are present), draw all boids
  for (int i = 0; i < zones.length; i++) {
    for (int j = 0; j < zones[i].length; j++) {  
        
      // draw walls since wall array lenghts are same as zone array lengths
      float wallX = i*zoneLength;
      float wallY = j*zoneLength;
      float wallWidth = 1.5 / 55 * zoneLength;
        
      noStroke();
      fill(foreground);
      if (verticalWalls[i][j] == 1)
        rect(wallX - wallWidth / 2, wallY - wallWidth / 2, wallWidth, zoneLength + wallWidth);
      else if (verticalWalls[i][j] == 2)
        rect(wallX - wallWidth / 2, wallY - wallWidth / 2, wallWidth, (zoneLength + wallWidth)/2);
      else if (verticalWalls[i][j] == 3)
        rect(wallX - wallWidth / 2, wallY + zoneLength/2 - wallWidth / 2, wallWidth, (zoneLength + wallWidth)/2);
        
      if (horizontalWalls[i][j] == 1)
        rect(wallX - wallWidth / 2, wallY - wallWidth / 2, zoneLength + wallWidth, wallWidth);
      else if (horizontalWalls[i][j] == 2)
        rect(wallX - wallWidth / 2, wallY - wallWidth / 2, (zoneLength + wallWidth)/2, wallWidth);
      else if (horizontalWalls[i][j] == 3)
        rect(wallX + zoneLength/2 - wallWidth / 2, wallY - wallWidth / 2, (zoneLength + wallWidth)/2, wallWidth);
        
        
      // create zones
      if (zones[i][j].size() > 0) {
        fill(255, 5);
      } else {
        noFill();
      }
      noStroke();
      rect(i * zoneLength, j * zoneLength, zoneLength, zoneLength);
        
      // draw boids
      if (!start) {
        for (int k = 0; k < zones[i][j].size(); k++) {
          Boid b = zones[i][j].get(k);
            
          // if it is block 4 and boid is in first square, remove
          if (b.position.x < 0 || b.position.x > width || b.position.y < 0 || b.position.y > height/2 && block == 4)
            zones[i][j].remove(b);
          
          b.display();
          if (!pause) b.move();
          
          // if boid crosses into another zone, switch boid into that zone's arraylist and remove from previous
          int zoneX = floor(b.position.x / zoneLength);
          int zoneY = floor(b.position.y / zoneLength);
          if (zoneX != i || zoneY != j) {
            try {
              zones[zoneX][zoneY].add(new Boid(
                                               b.position, 
                                               b.velocity, 
                                               b.condition, 
                                               b.type, 
                                               b.wearingMask, 
                                               b.route, 
                                               b.path, 
                                               b.pathIndex, 
                                               b.foundSeat, 
                                               b.seatPos, 
                                               b.index, 
                                               b.interactions,
                                               b.selected,
                                               b.sex,
                                               b.year,
                                               b.timeWithInfected
                                              ));
              zones[i][j].remove(b);
            } catch (Exception ie) {
              continue;
            }
          } else {
            if (block != 3) avgNumContacts += b.interactions.size()/float(numBoids);
          }
        }
      }
    }
  }
  
  if (editing <= 1 || !start) {
    
    // conversion to seconds, minutes, hours
    String time = convertTime(floor(totalSeconds) + 7*60*60 + 10*60);
    
    textSize(15);
    
    pushMatrix();
    if (start) translate(width/4 + spacingLength, height/2);
    else translate(width/2, height/2);
    
    pushMatrix();
    float rowSize = 0;
    for (float i = 0; i < 14; i++) {
      translate(0, rowSize);
      if (i < 10) {
        rowSize = spacingLength/2;
      } else {
        rowSize = spacingLength;
      }
      fill(30 + 10 * ((i+1)%2));
      rect(0, 0, width/4 - spacingLength/2, rowSize);
    }
    popMatrix();
    
    translate(spacingLength/8, -spacingLength/3);
    fill(foreground);
    textAlign(CENTER, CENTER);
    if (!start) text(time + " AM", width/8 - spacingLength/4, 0);
    else text("Last Simulation | Finished at: " + time + " AM", width/8 - spacingLength/4, 0);
    textAlign(LEFT, CENTER);
    text("% Chance of Transmission", 0, spacingLength/2);
    text("Block", 0, spacingLength);
    text("Number of Students", 0, spacingLength*1.5);
    text("Number of Infected Students", 0, spacingLength*2);
    text("Number of Teachers", 0, spacingLength*2.5);
    text("% Wearing Masks", 0, spacingLength*3);
    text("% Immune", 0, spacingLength*3.5);
    text("Social Distancing Factor", 0, spacingLength*4);
    text("Social Distancing Range", 0, spacingLength*4.5);
    text("Circular Hall Pattern", 0, spacingLength*5);
    
    text("Average Number of Contacts", 0, spacingLength*5.75);
    text("Total Time of\nInteraction", 0, spacingLength*6.75);
    text("Total Time of\nInteraction with Infected", 0, spacingLength*7.75);
    text("Total Weighted\nInteraction", 0, spacingLength*8.75);
    
    translate(width/4 - spacingLength*3/4, 0);
    textAlign(RIGHT, CENTER);
    text(nf(transmissionProbability * 100, 0, 3) + "%", 0, spacingLength/2);
    text(block+1, 0, spacingLength);
    text(numBoids, 0, spacingLength*1.5);
    text(numInfected, 0, spacingLength*2);
    text(classrooms.size(), 0, spacingLength*2.5);
    text(100*numWearingMasks/numBoids + "%", 0, spacingLength*3);
    text(100*numImmune/numBoids + "%", 0, spacingLength*3.5);
    text(socialDistancingFactor, 0, spacingLength*4);
    text(socialDistancingRange + " m", 0, spacingLength*4.5);
    text(booleanText(circularHallPattern), 0, spacingLength*5);
    
    
    pushMatrix();
    if (detailedGraphs) {
      translate(-spacingLength, 0);
      drawGraph(anc_list, new PVector(spacingLength*0.25, spacingLength*5.45), new PVector(spacingLength*0.8, spacingLength*0.8), highest_anc, color(180, 50, 50));
      drawGraph(it_list, new PVector(spacingLength*0.25, spacingLength*6.45), new PVector(spacingLength*0.8, spacingLength*0.8), highest_it, color(180, 50, 50));
      drawGraph(iit_list, new PVector(spacingLength*0.25, spacingLength*7.45), new PVector(spacingLength*0.8, spacingLength*0.8), highest_iit, color(180, 50, 50));
      drawGraph(wit_list, new PVector(spacingLength*0.25, spacingLength*8.45), new PVector(spacingLength*0.8, spacingLength*0.8), highest_wit, color(180, 50, 50));
    }
    
    fill(foreground);
    text(nf(avgNumContacts, 0, 3), 0, spacingLength*5.75);
    text(convertTime(floor(totalInteractionTime)), 0, spacingLength*6.75);
    text(convertTime(floor(totalInfectedInteractionTime)), 0, spacingLength*7.75);
    text(nf(totalWeightedInteractionTime, 0, 3), 0, spacingLength*8.75);
    popMatrix();
    
    popMatrix();
  }
  
  // reset mouse clicked variable
  isMouseClicked = false;
}
