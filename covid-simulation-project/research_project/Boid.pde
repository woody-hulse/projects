class Boid {
  
  
  // determines the state (susceptible, infected, dead, exposed, asymptomatic, or recovered) of the boid
  String condition;
  
  // determines the type (student, faculty, staff) of the boid
  String type;
  
  // is the boid's information being displayed (shown when boid is clicked)
  boolean selected;
  
  float size;
  float speed;
  color col;
  
  float timeWithInfected;
  
  // physics attributes
  PVector position;
  PVector velocity;
  PVector targetVelocity;
  PVector acceleration;
  
  // characteristics
  int age;
  int sex;
  String year;
  
  float turnSpeed;
  // how much boid avoids other boids
  float avoidance;
  // range at which boids avoid other boids
  float avoidanceRange;
  // range at which boids will avoid walls
  float wallAvoidanceRange;
  
  // range over which the virus can transmit
  float transmissionRange;
  // is subject wearing mask
  boolean wearingMask;
  
  // path to target
  int[][] path;
  int[][][] route;
  int pathIndex;
  
  // determine what side of the hall boid targets
  boolean travellingRight = true;
  
  // -1. 0, 1, 2, 3 based on dominant velocity direction
  int targetDirection = 0;
  
  // is boid in a seat
  boolean foundSeat;
  PVector seatPos;
  
  // new avoidance factor
  float avoidanceFactor = 0.004;
  
  // index of boid
  int index;
  
  // index of classroom
  int roomIndex = -1;
  
  // indexes of boids interacted with
  IntList interactions;
  
  Boid(
       PVector position, 
       PVector velocity, 
       String condition, 
       String type, 
       boolean wearingMask, 
       int[][][] route, 
       int[][] path, 
       int pathIndex, 
       boolean foundSeat, 
       PVector seatPos, 
       int index, 
       IntList interactions,
       boolean selected,
       int sex,
       String year,
       float timeWithInfected
      ) {
    
    this.position = position;
    this.velocity = velocity;
    targetVelocity = velocity;
    this.condition = condition;
    this.type = type;
    this.wearingMask = wearingMask;
    
    this.route = route;
    this.path = path;
    this.pathIndex = pathIndex;
    
    this.foundSeat = foundSeat;
    this.seatPos = seatPos;
    
    this.index = index;
    this.interactions = interactions;
    
    this.selected = selected;
    
    this.sex = sex;
    this.year = year;
    
    this.timeWithInfected = timeWithInfected;
    
    speed = 1.4 / scale * tickLength * tickSpeed;
    size = 8f / 55 * zoneLength;
    
    turnSpeed = .1;
    avoidance = speed * socialDistancingFactor;
    // 6 feet social distancing
    //avoidanceRange = zoneLength / 3.3 * 1.8288;
    avoidanceRange = socialDistancingRange / scale;
    
    transmissionRange = 2 / scale;
    
    wallAvoidanceRange = 20;
    
    col = color(107, 159, 242);
  }
  
  int[][] removeFirstElement(int[][] arr) {
    int[][] newArr = new int[arr.length-1][arr[0].length];
    
    for (int i = 1; i < arr.length; i++) {
      newArr[i-1] = arr[i];
    }
    return newArr;
  }
  
  void turn() {
   if (path.length > 0) {
     path = removeFirstElement(path);
   } else {
     // if boid is at last leg, then boid can be deleted
     // otherwise, direct boid toward its seat
   }
   
   if (path.length >= 2) {
     if (path[0][0] > path[1][0]) {
       travellingRight = false;
     } else {
       travellingRight = true;
     }
   }
  }
  
  PVector socialDistance(PVector position1, PVector position2, float distance) {
    // get vector pointing away from other boid
    PVector awayVector = PVector.sub(position1, position2);
    // normalize vector
    awayVector.normalize();
    // adjust for distance
    awayVector.div(distance);
    // multiply by avoidance constant
    awayVector.mult(avoidance);
    // no awayvector if teacher
    if (type == "Teacher")
      awayVector.mult(0);
    
    // dont let boid get pushed into wrong cells
    if (targetDirection == 2 || targetDirection == 3) {
      awayVector.x *= 0.5;
      awayVector.y *= 1.3;
      
      if (floor((position.x + awayVector.x) / zoneLength) != floor(position.x / zoneLength))
        awayVector.x = 0;
    } else if (targetDirection != -1) {
      awayVector.y *= 0.5;
      awayVector.x *= 1.3;
      
      if (floor((position.y + awayVector.y) / zoneLength) != floor(position.y / zoneLength))
        awayVector.y = 0;
    }
    
    awayVector.mult(tickSpeed);
    
    return awayVector;
  }
  
  // based on distances between boids, transmission probability is taken and added to total
  float calculateTransmissionProbability(float distance, Boid b2, boolean real) {

    // (Rh * fvh * fmh * fah) * (fat * fvv) * (fis * fms * Ts)
    float probability = 1;
    
    if (condition == "Immune" || b2.condition == "Immune")
      probability *= 0.08;
    
    // Rh
    probability *= lerp(80, 200, velocity.mag());
    // fvh
    probability *= lerp(.37, 0.0037, distance / transmissionRange);
    // fmh
    if (wearingMask)
      probability *= 0.37;
    // fah
    probability *= lerp(1, 0.1, distance / transmissionRange);
    // fat
    probability *= (1 / (distance / transmissionRange * 2 * PI));
    // fvv (environmental factors)
    probability *= 0.95;
    // fis (accounted for in fah)
    probability *= 1;
    // fms (chance of transmission through the average mask is 37%)
    if (b2.wearingMask)
      probability *= 0.37;
    // Ts (multiply by the speed and conversion rate of ticks in simulation)
    probability *= tickSpeed * tickLength;
    
    // NID (50% chance of infection with 10^5 virion load, per virion chance is 5*10^4)
    probability /= 50000;
    
    // change transmission probability (chance of both previous transmission and this transmission not occuring)
    if (real) transmissionProbability = 1 - (1 - transmissionProbability) * (1 - probability);
    
    return probability;
  }
  
  float calculateAerosolTransmissionProbability(int numInfected, float percentageWindows, boolean real) {
    
    float probability = 1;
    
    if (condition == "Immune")
      probability *= 0.92;
    
    // Rh (number of emittors at rest times pct of aerisolized particles
    probability *= numInfected * 50 * 0.06;

    // fvh
    probability *= 0.0037;
    // fmh (higher mask transmission for smaller particles)
    if (wearingMask)
      probability *= 0.6;
    // fvv
    probability *= (1 - percentageWindows/2);
    // fms
    if (wearingMask)
      probability *= 0.6;
    // Ts
    probability *= tickSpeed * tickLength;
    
    // NID (50% chance of infection with 10^6 virion load, per virion chance is 5*10^4)
    probability /= 50000;
    
    // change transmission probability (chance of both previous transmission and this transmission not occuring)
    if (real) {
      transmissionProbability = 1 - (1 - transmissionProbability) * (1 - probability);
      ap = 1 - (1 - ap) * (1 - probability);
    }
    
    return probability;
  }
  
  void findRoom() {
    int zoneX = floor(position.x / zoneLength);
    int zoneY = floor(position.y / zoneLength);
    
    roomIndex = -1;
    
    for (int i = 0; i < classrooms.size(); i++) {
      Classroom c = classrooms.get(i);
      if (c.x <= zoneX && c.x+c.l > zoneX && c.y <= zoneY && c.y+c.h > zoneY)
        roomIndex = i;
    }
  }

  void move() {
    
    // if block has changed, set new path
    if (pathIndex != block && !pause) {
      path = route[block];
      pathIndex = block;
    }
        
    int zoneX = floor(position.x / zoneLength);
    int zoneY = floor(position.y / zoneLength);
    
    // boid steers away from other boids
    for (int i = -1; i <= 1; i++) {
      for (int j = -1; j <= 1; j++) {
        try {
          for (Boid b : zones[zoneX + i][zoneY + j]) {
            float distance = dist(b.position.x, b.position.y, position.x, position.y);
              
            // if other boid is within range, will move away to socially distance
            if (distance < transmissionRange && distance > 0) {
              
              // if boid isnt in list of contacted boids, append
              if (!inList(interactions, b.index)) {
                interactions.append(b.index);
              }
              
              float chanceOfTransmission = calculateTransmissionProbability(distance, b, (condition == "Infected"));
              
              // count to total number of interactions
              totalInteractionTime += tickLength*tickSpeed;
              totalWeightedInteractionTime += chanceOfTransmission;
                  
              // draw line bewteen interacting boids
              stroke(50, 230, 180, 180);
              strokeWeight(0.5 * zoneLength / 55);
              line(position.x, position.y, b.position.x, b.position.y); // potentially change this section to creating a single away vector instead of one for each boid b
                  
              // calculate transmission probability
              if (condition == "Infected") {
                totalWeightedInfectedInteractionTime += chanceOfTransmission;
                b.timeWithInfected += chanceOfTransmission;
                totalInfectedInteractionTime += tickLength*tickSpeed;
              }
                  
              // add social distance vector to velocity and re-normalize
              if (!foundSeat || type == "Teacher") {
                if (distance < avoidanceRange)
                  position.add(socialDistance(position, b.position, distance).mult(3));
              }
            }
                
            // if boid is within range of transmission, will add to the total transmission probability
          }
            
          // wall avoidance
          if (path.length > 0) {
            // avoid vertical walls, only need to check wall of that cell and wall of cell n + 1
            if (i >= 0 && j == 0) {
              if (verticalWalls[zoneX + i][zoneY + j] == 1) {
                      
                // calculate distance between boid and wall
                float distance = position.x - (zoneX + i) * zoneLength;
                    
                if (abs(distance) < wallAvoidanceRange) {
                  // draw line bewteen boid and wall
                  // stroke(80);
                  // line(position.x, position.y, (zoneX + i) * zoneLength, position.y);
                  position.add(new PVector(wallAvoidanceRange/(distance+0.5), 0).mult(avoidanceFactor));
                }
              }
              
              // dont let boid get pushed into cells it doesn't want to go to
              if (abs(velocity.y*0.8) > abs(velocity.x)) {
                if (velocity.x < 0 && position.x > (zoneX + i) * zoneLength) {
                  if (position.x + velocity.x < (zoneX + i) * zoneLength) {
                    position.x += 1;
                  }
                } else if (velocity.x > 0 && position.x < (zoneX + i) * zoneLength) {
                  if (position.x + velocity.x > (zoneX + i) * zoneLength) {
                    position.x -= 1;
                  }
                }
              }
            }
                
            // avoid horizontal walls
            if (i == 0 && j >= 0) {
              if (horizontalWalls[zoneX + i][zoneY + j] == 1) {
                      
                // calculate distance between boid and wall
                float distance = position.y - (zoneY + j) * zoneLength;
                    
                if (abs(distance) < wallAvoidanceRange) {
                  position.add(new PVector(0, wallAvoidanceRange/(distance+0.5)).mult(avoidanceFactor));
                }
              }
              
              // dont let boid get pushed into cells it doesn't want to go to
              if (abs(velocity.x*0.8) > abs(velocity.y)) {
                if (velocity.y < 0 && position.y > (zoneY + j) * zoneLength) {
                  if (position.y + velocity.y < (zoneY + j) * zoneLength) {
                    position.y += 1;
                  }
                } else if (velocity.y > 0 && position.y < (zoneY + j) * zoneLength) {
                  if (position.y + velocity.y > (zoneY + j) * zoneLength) {
                    position.y -= 1;
                  }
                }
              }
            }
          }
        } catch(Exception ie) {
          continue;
        }
      }
    }
    
    targetDirection = -1;
    
    // while the boid is more than sqrt(2) pixels away from target, move toward target.
    if (path.length > 0 && type != "Teacher") {
      foundSeat = false;
      roomIndex = -1;
        
      if (zoneX != path[0][0] || zoneY != path[0][1]) {
        // boid steeers toward target
        PVector targetPoint;
        float side;
        if (velocity.x > 0) side = 0.75;
        else side = 0.25;
        targetPoint = new PVector((path[0][0]+0.5)*zoneLength, (path[0][1]+side)*zoneLength);
        if (zoneY == 4 && path[0][1] == 4&& circularHallPattern) position.y += ((4 + side)*zoneLength - position.y) / zoneLength;
        
        // determine target direction
        PVector directionVector = PVector.sub(targetPoint, position).normalize();
        if (abs(directionVector.x) > 0.6) {
          if (directionVector.x > 0) {
            targetDirection = 0;
          } else { 
            targetDirection = 1;
          }
        } else if (abs(directionVector.y) > 0.6) {
          if (directionVector.y > 0) {
            targetDirection = 2;
          } else {
            targetDirection = 3;
          }
        }
          
        targetVelocity = new PVector(targetPoint.x - position.x, targetPoint.y - position.y).normalize();
        // additional multiplicaiton step included to arbitrarily increase desire of boid to reach its target
        velocity.add(targetVelocity.sub(velocity).mult(1.6).normalize().mult(turnSpeed));
                
       // ensure that boids stay within model'
       if (position.x < 0) position.x = 1;
       if (position.x > zoneLength*zones.length) position.x = zoneLength*zones.length - 1;
       if (position.y < 0) position.y = zoneLength + 1;
       if (position.y > zoneLength*zones[0].length) position.y = zoneLength*zones[0].length - 1;
          
      //position.add(PVector.mult(velocity, speed));
      } else {
        turn();
      }
    } else if (!foundSeat){
    // find seat
        
      findRoom();
      
      if (roomIndex >= 0) {
        
        Classroom c = classrooms.get(roomIndex);
        // first element is seat index, second is number of adjacent open seats
        int[] bestSeat = {0, 0};
        for (int i = 0; i < c.seats.size(); i++) {
          
          if (c.seats.get(i).open) {
            int numAdjacent = 0;
            
            for (int u = -1; u <= 1; u++) {
              for (int l = -1; l <= 1; l++) {
                if (u == 0 || l == 0) {
                  try {
                    if (c.seats.get(i + u + l*c.numInCol).open)
                      numAdjacent++;
                  } catch (Exception ie) {
                    numAdjacent++;
                  }
                }
              }
            }
            
            if (numAdjacent > bestSeat[1]) {
              int[] newBest = {i, numAdjacent};
              bestSeat = newBest;
            }
            
            if (numAdjacent == bestSeat[1]) { /// && random(0, 1) < 0.2
              int[] newBest = {i, numAdjacent};
              bestSeat = newBest;
            }
          }
        }
        Seat chosen = c.seats.get(bestSeat[0]);
        if (condition == "Infected")
          chosen.containsInfected = true;
        chosen.open = false;
        foundSeat = true;
        seatPos = new PVector(chosen.x*zoneLength, chosen.y*zoneLength);
      }
    } else if (dist(position.x, position.y, seatPos.x, seatPos.y) > 8f * zoneLength / 55f) {
      targetVelocity = new PVector(seatPos.x - position.x, seatPos.y - position.y).normalize();
      velocity.add(targetVelocity.sub(velocity).normalize().mult(turnSpeed));
    }
    
    if (foundSeat && dist(position.x, position.y, seatPos.x, seatPos.y) < 8f * zoneLength / 55f) {
      // get class and compute airborne transmission
      if (condition == "Susceptible") {
        if (roomIndex != -1) {
          Classroom c = classrooms.get(roomIndex);
          timeWithInfected += calculateAerosolTransmissionProbability(c.infectedCount, c.percentageWindows, true) * 10;
          totalWeightedInteractionTime += calculateAerosolTransmissionProbability(1, c.percentageWindows, false);
        } else {
          findRoom();
        }
      }
      // lock boid into seat, forward facing
      position = seatPos;
      velocity = new PVector(.1, 0.002);
      if (type != "Teacher")
        numSeated ++;
    }
      
    if ((dist(position.x, position.y, seatPos.x, seatPos.y) > 8f * zoneLength / 55f || !foundSeat) && type != "Teacher")
      position.add(PVector.mult(velocity, speed));
  }
  
  void display() {
    
    // draw lines if paused
    if (pause) {
      int zoneX = floor(position.x / zoneLength);
      int zoneY = floor(position.y / zoneLength);
    
      for (int i = -1; i <= 1; i++) {
        for (int j = -1; j <= 1; j++) {
          try {
            for (Boid b : zones[zoneX + i][zoneY + j]) {
              float distance = dist(b.position.x, b.position.y, position.x, position.y);
              
              // if other boid is within range, will move away to socially distance
              if (distance < avoidanceRange && distance > 0) {
                
              // draw line bewteen interacting boids
              stroke(50, 230, 180, 180);
              strokeWeight(0.5 * zoneLength / 55);
              line(position.x, position.y, b.position.x, b.position.y);
              }
            }
          } catch(Exception ie) {
            continue;
          }
        }
      }
    }
    
    // draw boid
    pushMatrix();
    translate(position.x, position.y);
    
    noStroke();
    fill(100, 230, 180, 20);
    if (selected) {
      //ellipse(0, 0, avoidanceRange*2, avoidanceRange*2);
      stroke(foreground);
      noFill();
      strokeWeight(2);
      rect(-size*0.8, -size, size*2, size*2);
      noStroke();
    }
    if (condition == "Infected")
      fill(200, 120, 50);
    else {
      float cr = pow(25, -25*timeWithInfected);
      fill(107 * (2 - cr), 159 * cr, 242 * cr);
    }
    pushMatrix();
    if (type == "Teacher") { 
      rotate(-PI);
      fill(130, 40, 180);
    } else rotate(PVector.angleBetween(velocity, new PVector(1, 0)) * velocity.y / abs(velocity.y));
        
    createShape();
    beginShape();
    vertex(-size/2, size/2);
    vertex(size, 0);
    vertex(-size/2, -size/2);
    vertex(-size/4, 0);
    vertex(-size/2, size/2);
    endShape();
    
    popMatrix();
    
    popMatrix();
  }
  
}
