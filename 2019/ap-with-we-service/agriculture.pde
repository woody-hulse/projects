PImage climate;
PImage soil;
PImage ph;
PImage world;

PFont font;

color a = color(50, 150, 255);
color b = color(0, 0, 255);

int phoffset = 14;

int window_size = 15;

  //0; rice
  //1: corn
  //2: soy
  //3: wheat
  //4: potato
  //5: tomato
  //6: sugarcane
  //7: cattle
  //8: vegetable
  //9: apple
  //10: banana
  //11: mango
  //12: sheep
  //13: onion
  //14: bean
  //15: peanut
  //16: olive
  //17: rapeseed
  //18: almond
  //19: walnuts
boolean[] food = new boolean[20];

int[] zoom = new int[2];

void setup () {
  size(1680, 1000);
  
  climate = loadImage("climate_zones.png");
  soil = loadImage("soil_types.png");
  ph = loadImage("soil_ph.png");
  world = loadImage("world.png");
  
  font = loadFont("Futura-MediumItalic-48.vlw");
  
  background(0);
  
  loadPixels();
  climate.loadPixels();
  for (int x = 7; x < 581; x++) {
    for (int y = 17; y < 313; y++) {
      int loc = x+y*602;
      
      float r = red(climate.pixels[loc]);
      float g = green(climate.pixels[loc]);
      float b = blue(climate.pixels[loc]);
      
      pixels[floor((560*(x+floor(y/1.03)*width*574/560))/574)-16*width-6] = color(r, g, b);
    }
  }
  updatePixels();
  
  soil.loadPixels();
  for (int x = 0; x < 560; x++) {
    for (int y = 37; y < 289; y++) {
      int loc = x+y*560;
      
      float r = red(soil.pixels[loc]);
      float g = green(soil.pixels[loc]);
      float b = blue(soil.pixels[loc]);
      
      pixels[floor((x+y*width))-(37*width)+1120] = color(r, g, b);
    }
  }
  updatePixels();
  
  ph.loadPixels();
  for (int x = 44; x < 1133; x++) {
    for (int y = 28; y < 575; y++) {
      int loc = x+y*1200;
      
      float r = red(ph.pixels[loc]);
      float g = green(ph.pixels[loc]);
      float b = blue(ph.pixels[loc]);
      
      pixels[floor((560*(x+floor(y/1.9)*width*1089/560))/1089)-14*width+538+phoffset] = color(r, g, b);
    }
  }
  updatePixels();
  
  for (int x = 0; x < 560; x++) {
    for (int y = 0; y < 313; y++) {
      
      float r1 = red(pixels[x+y*width]);
      float g1 = green(pixels[x+y*width]);
      float b1 = blue(pixels[x+y*width]);
      
      float r2 = red(pixels[x+y*width+560]);
      float g2 = green(pixels[x+y*width+560]);
      float b2 = blue(pixels[x+y*width+560]);
      
      float r3 = red(pixels[x+y*width+1120]);
      float g3 = green(pixels[x+y*width+1120]);
      float b3 = blue(pixels[x+y*width+1120]);
      
        //neutral soil
      if (r2 > b2 && g2 > b2) pixels[313*width + x+y*width + 560] = color(255);
        //acidic soil
      if (r2 > b2 && r2 > g2) pixels[313*width + x+y*width + 560] = color(205);
        //basic soil (alkaline)
      if (b2 > r2 && b2 > g2) pixels[313*width + x+y*width + 560] = color(155);
      
        //alfisols
      if (r3 >= 120 && r3 <= 200 && g3 >= 180 && g3 <= 215 && b3 >= 60 && b3 <= 145) pixels[x+y*width + 1120 + 313*width] = color(255);
        //aridisols
      if (r3 >= 200 && r3 <= 255 && g3 >= 210 && g3 <= 250 && b3 >= 165 && b3 <= 210) pixels[x+y*width + 1120 + 313*width] = color(240);
        //gelisols
      if (r3 >= 140 && r3 <= 180 && g3 >= 165 && g3 <= 200 && b3 >= 150 && b3 <= 225) {
        if ((r3 + b3 + g3)/(3*b3) < 0.98 || (r3 + b3 + g3)/(3*b3) > 1.02)
          pixels[x+y*width + 1120 + 313*width] = color(225);
      }
        //entisols
      if (r3 >= 130 && r3 <= 185 && g3 >= 175 && g3 <= 220 && b3 >= 150 && b3 <= 195) {
        if ((r3 + b3 + g3)/(3*g3) < 0.95 || (r3 + b3 + g3)/(3*g3) > 1.05)
          pixels[x+y*width + 1120 + 313*width] = color(210);
      }
        //histosols
      if (r3 >= 110 && r3 <= 150 && g3 >= 40 && g3 <= 90 && b3 >= 25 && b3 <= 80) pixels[x+y*width + 1120 + 313*width] = color(195);
        //inceptisols
      if (r3 >= 190 && r3 <= 255 && g3 >= 150 && g3 <= 180 && b3 >= 0 && b3 <= 140) pixels[x+y*width + 1120 + 313*width] = color(180);
        //mollisols
      if (r3 >= 60 && r3 <= 120 && g3 >= 140 && g3 <= 180 && b3 >= 50 && b3 <= 100) pixels[x+y*width + 1120 + 313*width] = color(165);
        //oxisols
      if (r3 >= 185 && r3 <= 255 && g3 >= 100 && g3 <= 160 && b3 >= 70 && b3 <= 131) pixels[x+y*width + 1120 + 313*width] = color(150);
        //spodosols
      if (r3 >= 180 && r3 <= 210 && g3 >= 145 && g3 <= 165 && b3 >= 160 && b3 <= 205) pixels[x+y*width + 1120 + 313*width] = color(135);
        //ultisols
      if (r3 >= 190 && r3 <= 255 && g3 >= 175 && g3 <= 240 && b3 >= 30 && b3 <= 140) pixels[x+y*width + 1120 + 313*width] = color(120);
        //vertisols
      if (r3 >= 70 && r3 <= 105 && g3 >= 70 && g3 <= 110 && b3 >= 110 && b3 <= 160) pixels[x+y*width + 1120 + 313*width] = color(105);
        //rocky land
      if (r3 >= 190 && r3 <= 220 && g3 >= 190 && g3 <= 215 && b3 >= 170 && b3 <= 205) pixels[x+y*width + 1120 + 313*width] = color(90);
        //shifting sand
      if (r3 >= 180 && r3 <= 220 && g3 >= 175 && g3 <= 220 && b3 >= 170 && b3 <= 210) pixels[x+y*width + 1120 + 313*width] = color(75);
        
      color c2 = pixels[x+y*width+560];
        
      if (c2 != color(255) && c2 != color(0) && (r2+b2+g2)/3 != r2) {
          //tundra
        if (r1 >= 135 && r1 <= 165 && g1 >= 95 && g1 <= 145 && b1 >= 155 && b1 <= 195) pixels[313*width + x+y*width] = color(255);
          //subartic
        if (r1 >= 165 && r1 <= 185 && g1 >= 170 && g1 <= 205 && b1 >= 215 && b1 <= 250) pixels[313*width + x+y*width] = color(240);
          //cooler summer
        if (r1 >= 115 && r1 <= 145 && g1 >= 145 && g1 <= 180 && b1 >= 195 && b1 <= 230) pixels[313*width + x+y*width] = color(225);
          //cool summer
        if (r1 >= 75 && r1 <= 110 && g1 >= 90 && g1 <= 125 && b1 >= 120 && b1 <= 155) pixels[313*width + x+y*width] = color(210);
          //highlands
        if (r1 >= 155 && r1 <= 190 && g1 >= 135 && g1 <= 165 && b1 >= 80 && b1 <= 125) pixels[313*width + x+y*width] = color(195);
          //tropical wet
        if (r1 >= 80 && r1 <= 115 && g1 >= 175 && g1 <= 200 && b1 >= 55 && b1 <= 115) pixels[313*width + x+y*width] = color(180);
          //tropical dry
        if (r1 >= 155 && r1 <= 220 && g1 >= 195 && g1 <= 230 && b1 >= 95 && b1 <= 165) pixels[313*width + x+y*width] = color(165);
          //semiarid
        if (r1 >= 235 && r1 <= 255 && g1 >= 225 && g1 <= 255 && b1 >= 115 && b1 <= 155) pixels[313*width + x+y*width] = color(150);
          //arid
        if (r1 >= 220 && r1 <= 255 && g1 >= 145 && g1 <= 170 && b1 >= 0 && b1 <= 65) pixels[313*width + x+y*width] = color(135);
          //mediterranean
        if (r1 >= 110 && r1 <= 140 && g1 >= 125 && g1 <= 165 && b1 >= 145 && b1 <= 200) pixels[313*width + x+y*width] = color(120);
          //marine west coast
        if (r1 >= 140 && r1 <= 175 && g1 >= 155 && g1 <= 180 && b1 >= 90 && b1 <= 135) pixels[313*width + x+y*width] = color(105);
          //humid subtropical
        if (r1 >= 215 && r1 <= 235 && g1 >= 210 && g1 <= 240 && b1 >= 165 && b1 <= 185) pixels[313*width + x+y*width] = color(90);
      }
    }
  }
}

void draw () {
  
  world.loadPixels();
  for (int x = 76; x < 1033; x++) {
    for (int y = 24; y < 509; y++) {
      int loc = x+y*1108;
      
      float r = red(world.pixels[loc]);
      float g = green(world.pixels[loc]);
      float b = blue(world.pixels[loc]);
      
      pixels[floor((560*(x+floor(y/1.68)*width*957/560))/957)-24*width+636*width-44] = color(r, g, b);
    }
  }
  updatePixels();
  
  for (int x = 0; x < 560; x++) {
    for (int y = 0; y < 313; y++) {
      
        //potatoes
      if(food[4] == true){
        if (pixels[313*width + x+y*width + 560] == color(205) || pixels[313*width + x+y*width + 560] == color(255)) {
          if (pixels[313*width + x+y*width] == color(225) || pixels[313*width + x+y*width] == color(210) || pixels[313*width + x+y*width] == color(105)){
          
            pixels[x+y*width + 626*width] = a;
          
            if (pixels[x+y*width + 1120 + 313*width] == color(180) || pixels[x+y*width + 1120 + 313*width] == color(195) || pixels[x+y*width + 1120 + 313*width] == color(255))
              pixels[x+y*width + 626*width] = b;
          }
        }
      }
      
        //wheat
      if(food[3] == true){
        if (pixels[313*width + x+y*width + 560] == color(205) || pixels[313*width + x+y*width + 560] == color(255)) {
          if (pixels[313*width + x+y*width] == color(210) || pixels[313*width + x+y*width] == color(150) || pixels[313*width + x+y*width] == color(165) || pixels[313*width + x+y*width] == color(105)) {
          
            pixels[x+y*width + 626*width] = a;
          
            if (pixels[x+y*width + 1120 + 313*width] == color(255) || pixels[x+y*width + 1120 + 313*width] == color(165) || pixels[x+y*width + 1120 + 313*width] == color(195) || pixels[x+y*width + 1120 + 313*width] == color(180) || pixels[x+y*width + 1120 + 313*width] == color(135)) 
              pixels[x+y*width + 626*width] = color(0, 0, 255);
          }
        }
      }
      
        //corn
      if (food[1] == true){
        if (pixels[313*width + x+y*width + 560] == color(205) || pixels[313*width + x+y*width + 560] == color(255)) {
          if (pixels[313*width + x+y*width] == color(225) || pixels[313*width + x+y*width] == color(210) || pixels[313*width + x+y*width] == color(165) || pixels[313*width + x+y*width] == color(90) || pixels[313*width + x+y*width] == color(105)) {
            
            pixels[x+y*width + 626*width] = a;
            
            if (pixels[x+y*width + 1120 + 313*width] == color(255) || pixels[x+y*width + 1120 + 313*width] == color(135) || pixels[x+y*width + 1120 + 313*width] == color(180) || pixels[x+y*width + 1120 + 313*width] == color(195) || pixels[x+y*width + 1120 + 313*width] == color(165))
              pixels[x+y*width + 626*width]= b;
          }
        }
      }
      
        //rice
      if (food[0] == true) {
        if (pixels[313*width + x+y*width + 560] == color(205) || pixels[313*width + x+y*width + 560] == color(255)) {
          if (pixels[313*width + x+y*width] == color(150) || pixels[313*width + x+y*width] == color(210) || pixels[313*width + x+y*width] == color(195) || pixels[313*width + x+y*width] == color(180) || pixels[313*width + x+y*width] == color(90) || pixels[313*width + x+y*width] == color(120)) {
            
            pixels[x+y*width + 626*width] = a;
            
            if (pixels[x+y*width + 1120 + 313*width] == color(120) || pixels[x+y*width + 1120 + 313*width] == color(150) || pixels[x+y*width + 1120 + 313*width] == color(105) || pixels[x+y*width + 1120 + 313*width] == color(165) || pixels[x+y*width + 1120 + 313*width] == color(195) || pixels[x+y*width + 1120 + 313*width] == color(255))
              pixels[x+y*width + 626*width] = b;
          }
        }
      }
      
        //soybeans
      if (food[2] == true) {
        if (pixels[313*width + x+y*width + 560] == color(205) || pixels[313*width + x+y*width + 560] == color(255)) {
          if (pixels[313*width + x+y*width] == color(225) || pixels[313*width + x+y*width] == color(195) || pixels[313*width + x+y*width] == color(210) || pixels[313*width + x+y*width] == color(90) || pixels[313*width + x+y*width] == color(165) || pixels[313*width + x+y*width] == color(180) || pixels[313*width + x+y*width] == color(90) || pixels[313*width + x+y*width] == color(120)  || pixels[313*width + x+y*width] == color(105)) {
            
            pixels[x+y*width + 626*width] = a;
            
            if (pixels[x+y*width + 1120 + 313*width] == color(135) || pixels[x+y*width + 1120 + 313*width] == color(105) || pixels[x+y*width + 1120 + 313*width] == color(120) || pixels[x+y*width + 1120 + 313*width] == color(195) || pixels[x+y*width + 1120 + 313*width] == color(165) ||  pixels[x+y*width + 1120 + 313*width] == color(150))
              pixels[x+y*width + 626*width] = b;
          }
        }
      }
      
        //tomatoes
      if (food[5] == true) {
        if (pixels[313*width + x+y*width + 560] == color(205) || pixels[313*width + x+y*width + 560] == color(255)) {
          if (pixels[313*width + x+y*width] == color(210) || pixels[313*width + x+y*width] == color(150) || pixels[313*width + x+y*width] == color(165) || pixels[313*width + x+y*width] == color(90) || pixels[313*width + x+y*width] == color(120)) {
            
            pixels[x+y*width + 626*width] = a;
            
            if (pixels[x+y*width + 1120 + 313*width] == color(195) || pixels[x+y*width + 1120 + 313*width] == color(135) || pixels[x+y*width + 1120 + 313*width] == color(165) || pixels[x+y*width + 1120 + 313*width] == color(255) || pixels[x+y*width + 1120 + 313*width] == color(180))
              pixels[x+y*width + 626*width] = b;
          }
        }
      }
      
        //cattle
      if (food[7] == true) {
        if (pixels[313*width + x+y*width] == color(165) || pixels[313*width + x+y*width] == color(120) || pixels[313*width + x+y*width] == color(105) || pixels[313*width + x+y*width] == color(90) || pixels[313*width + x+y*width] == color(210))
          pixels[x+y*width + 626*width] = a;
      }
      
        //sugarcane
      if (food[6] == true) {
        if (pixels[313*width + x+y*width + 560] == color(205) || pixels[313*width + x+y*width + 560] == color(255)) {
          if (pixels[313*width + x+y*width] == color(180) || pixels[313*width + x+y*width] == color(165) || pixels[313*width + x+y*width] == color(90)) {
            
            pixels[x+y*width + 626*width] = a;
            
            if (pixels[x+y*width + 1120 + 313*width] == color(150) || pixels[x+y*width + 1120 + 313*width] == color(135))
              pixels[x+y*width + 626*width] = b;
          }
        }
      }
      
        //vegetables
      if (food[8] == true) {
        if (pixels[313*width + x+y*width + 560] == color(205) || pixels[313*width + x+y*width + 560] == color(255) || pixels[313*width + x+y*width + 560] == color(155)) {
          if (pixels[313*width + x+y*width] == color(210) || pixels[313*width + x+y*width] == color(180) || pixels[313*width + x+y*width] == color(165) || pixels[313*width + x+y*width] == color(150) || pixels[313*width + x+y*width] == color(120) || pixels[313*width + x+y*width] == color(105) || pixels[313*width + x+y*width] == color(90)) {
            
            pixels[x+y*width + 626*width] = a;
            
            if (pixels[x+y*width + 1120 + 313*width] == color(120) || pixels[x+y*width + 1120 + 313*width] == color(195) || pixels[x+y*width + 1120 + 313*width] == color(135) || pixels[x+y*width + 1120 + 313*width] == color(165) || pixels[x+y*width + 1120 + 313*width] == color(255) || pixels[x+y*width + 1120 + 313*width] == color(180))
              pixels[x+y*width + 626*width] = b;
          }
        }
      }
      
        //apples
      if (food[9] == true) {
        if (pixels[313*width + x+y*width + 560] == color(205) || pixels[313*width + x+y*width + 560] == color(255)) {
          if (pixels[313*width + x+y*width] == color(225) || pixels[313*width + x+y*width] == color(210) || pixels[313*width + x+y*width] == color(240) || pixels[313*width + x+y*width] == color(225) || pixels[313*width + x+y*width] == color(150) || pixels[313*width + x+y*width] == color(225)
               || pixels[313*width + x+y*width] == color(120) || pixels[313*width + x+y*width] == color(105) || pixels[313*width + x+y*width] == color(90)) {
                 
            pixels[x+y*width + 626*width] = a;
            
            if (pixels[x+y*width + 1120 + 313*width] == color(135) || pixels[x+y*width + 1120 + 313*width] == color(120) || pixels[x+y*width + 1120 + 313*width] == color(165) || pixels[x+y*width + 1120 + 313*width] == color(255)
                || pixels[x+y*width + 1120 + 313*width] == color(180)) pixels[x+y*width + 626*width] = b;
                
          }
        }
      }
      
        //bananas
      if (food[10] == true) {
        if (pixels[313*width + x+y*width + 560] == color(205)) {
          if (pixels[313*width + x+y*width] == color(180) || pixels[313*width + x+y*width] == color(165)) {
            
            pixels[x+y*width + 626*width] = a;
            
            if (pixels[x+y*width + 1120 + 313*width] == color(120) || pixels[x+y*width + 1120 + 313*width] == color(255))
              pixels[x+y*width + 626*width] = b;
          }
        }
      }
      
        //mangoes
      if (food[11] == true) {
        if (pixels[313*width + x+y*width + 560] == color(205) || pixels[313*width + x+y*width + 560] == color(255)) {
          
        }
      }
      
        //sheep
      if (food[12] == true) {
        
      }
      
        //onions
      if (food[13] == true) {
        if (pixels[313*width + x+y*width + 560] == color(205)) {
          
        }
      }
      
        //beans
      if (food[14] == true) {
        if (pixels[313*width + x+y*width + 560] == color(205) || pixels[313*width + x+y*width + 560] == color(255)) {
          
        }
      }
      
        //peanuts
      if (food[15] == true) {
        if (pixels[313*width + x+y*width + 560] == color(205) || pixels[313*width + x+y*width + 560] == color(255)) {
          
        }
      }
      
        //olives
      if (food[16] == true) {
        if (pixels[313*width + x+y*width + 560] == color(205) || pixels[313*width + x+y*width + 560] == color(255) || pixels[313*width + x+y*width + 560] == color(155)) {
          
        }
      }
      
        //rapeseed
      if (food[17] == true) {
        if (pixels[313*width + x+y*width + 560] == color(205) || pixels[313*width + x+y*width + 560] == color(255)) {
          
        }
      }
      
        //almonds
      if (food[18] == true) {
        if (pixels[313*width + x+y*width + 560] == color(205) || pixels[313*width + x+y*width + 560] == color(255) || pixels[313*width + x+y*width + 560] == color(155)) {
          
        }
      }
      
        //walnuts
      if (food[19] == true) {
        if (pixels[313*width + x+y*width + 560] == color(205) || pixels[313*width + x+y*width + 560] == color(255)) {
          
        }
      }
    }
  }
  updatePixels();
  
  stroke(255);
  strokeWeight(2);
  line(0, 600, width, 600);
  noStroke();
  textFont(font);
  textSize(10);
  textAlign(LEFT, UP);

  fill(255);
  text("each pixel is approximately 160 square miles of land", 20, 615);
  
  fill(0);
  rect(560, 626, width-560, 400);
  fill(20);
  rect(570, 610, 580, 330);
  
  if (mousePressed) {
    // <= 105
    for (int x = 0; x < 7; x++) {
      for (int y = 0; y < 15; y++) {
        if (x + y*7 <= food.length) {
          if (mouseX > 585 + x*80 && mouseX < 645 + x*80 && mouseY > 613 + y*25 && mouseY < 628 + y*25) {
            for (int i = 0; i < food.length; i++) {
              if (i != x + y*7) 
                food[i] = false;
              else food[i] = true;
            }
          }
        }
      }
    }
  }
 
  fill(255);
  for (int x = 0; x < 7; x++) {
    for (int y = 0; y < 15; y++) {
      if (x + y*7 < food.length) {
        if (food[x + y*7] == true)
          ellipse(586 + x*80, 625 + y*15, 4, 4);
      }
    }
  }
 
  text("RICE", 590, 630);
  text("CORN", 670, 630);
  text("SOYBEAN", 750, 630);
  text("WHEAT", 830, 630);
  text("POTATO", 910, 630);
  text("TOMATO", 990, 630);
  text("SUGARCANE", 1070, 630);
  text("CATTLE", 590, 645);
  text("VEGETABLE", 670, 645);
  text("APPLE", 750, 645);
  text("BANANA", 830, 645);
  text("MANGO", 910, 645);
  text("SHEEP", 990, 645);
  text("ONION", 1070, 645);
  text("BEAN", 590, 660);
  text("PEANUT", 670, 660);
  text("OLIVE", 750, 660);
  text("RAPESEED", 830, 660);
  text("ALMOND", 910, 660);
  text("WALNUT", 990, 660);
  
  if (mouseX < 549 && mouseY > 637 && mouseY < 928) {
    textSize(15);
    fill(255);
    noStroke();
    fill(0, 60);
    rect(mouseX-window_size, mouseY-window_size, window_size*2, window_size*2);
    
    if(mousePressed) {
      zoom[0] = mouseX;
      zoom[1] = mouseY;
    }
  }
  
  if (zoom[0] != 0 && zoom[1] != 0) {
    
    //same key as food array
    int[] amount = new int[food.length];
    
    world.loadPixels();
    for (int x = 76+(int)((957*(zoom[0]-window_size))/560); x < 76+(int)((957*(zoom[0]+window_size))/560); x++) {
      for (int y = 24+(int)((zoom[1]-626-window_size)*1.68); y < 24+(int)((zoom[1]-626+window_size)*1.68); y++) {
        int loc = x+y*1108;
      
        float r = red(world.pixels[loc]);
        float g = green(world.pixels[loc]);
        float b = blue(world.pixels[loc]);
      
        pixels[x+y*width-24*width+636*width-44+560-zoom[0]*957/560-(zoom[1]-626)*width-floor((zoom[1]-626)/1.5)*width+x-floor(zoom[0]*1.7)+y*width-floor((zoom[1]-626)*1.7)*width] = color(r, g, b);
        pixels[x+y*width-24*width+636*width-44+560-zoom[0]*957/560-(zoom[1]-626)*width-floor((zoom[1]-626)/1.5)*width+x-floor(zoom[0]*1.7)+y*width-floor((zoom[1]-626)*1.7)*width+1] = color(r, g, b);
        pixels[x+y*width-24*width+636*width-44+560-zoom[0]*957/560-(zoom[1]-626)*width-floor((zoom[1]-626)/1.5)*width+x-floor(zoom[0]*1.7)+y*width-floor((zoom[1]-626)*1.7)*width+width] = color(r, g, b);
        pixels[x+y*width-24*width+636*width-44+560-zoom[0]*957/560-(zoom[1]-626)*width-floor((zoom[1]-626)/1.5)*width+x-floor(zoom[0]*1.7)+y*width-floor((zoom[1]-626)*1.7)*width+width+1] = color(r, g, b);
      }
    }
    updatePixels();
    updatePixels();
    
    for (int x = zoom[0] - window_size; x < zoom[0] + window_size; x ++) {
      for (int y = zoom[1] - window_size - 626; y < zoom[1] + window_size - 626; y++) {
        
        if (pixels[313*width + x+y*width + 560] == color(205) || pixels[313*width + x+y*width + 560] == color(255)) {
          if (pixels[313*width + x+y*width] == color(225) || pixels[313*width + x+y*width] == color(210) || pixels[313*width + x+y*width] == color(105)){
            if (pixels[x+y*width + 1120 + 313*width] == color(180) || pixels[x+y*width + 1120 + 313*width] == color(195) || pixels[x+y*width + 1120 + 313*width] == color(255))
              amount[4] ++;
          }
        }
      
      
        
        if (pixels[313*width + x+y*width + 560] == color(205) || pixels[313*width + x+y*width + 560] == color(255)) {
          if (pixels[313*width + x+y*width] == color(210) || pixels[313*width + x+y*width] == color(150) || pixels[313*width + x+y*width] == color(165) || pixels[313*width + x+y*width] == color(105)) {
            if (pixels[x+y*width + 1120 + 313*width] == color(255) || pixels[x+y*width + 1120 + 313*width] == color(165) || pixels[x+y*width + 1120 + 313*width] == color(195) || pixels[x+y*width + 1120 + 313*width] == color(180) || pixels[x+y*width + 1120 + 313*width] == color(135)) 
              amount[3] ++;
          }
        }
      
      
        
        if (pixels[313*width + x+y*width + 560] == color(205) || pixels[313*width + x+y*width + 560] == color(255)) {
          if (pixels[313*width + x+y*width] == color(225) || pixels[313*width + x+y*width] == color(210) || pixels[313*width + x+y*width] == color(165) || pixels[313*width + x+y*width] == color(90) || pixels[313*width + x+y*width] == color(105)) {
            if (pixels[x+y*width + 1120 + 313*width] == color(255) || pixels[x+y*width + 1120 + 313*width] == color(135) || pixels[x+y*width + 1120 + 313*width] == color(180) || pixels[x+y*width + 1120 + 313*width] == color(195) || pixels[x+y*width + 1120 + 313*width] == color(165))
              amount[1] ++;
          }
        }
      
      
        
        if (pixels[313*width + x+y*width + 560] == color(205) || pixels[313*width + x+y*width + 560] == color(255)) {
          if (pixels[313*width + x+y*width] == color(150) || pixels[313*width + x+y*width] == color(210) || pixels[313*width + x+y*width] == color(195) || pixels[313*width + x+y*width] == color(180) || pixels[313*width + x+y*width] == color(90) || pixels[313*width + x+y*width] == color(120)) {
            if (pixels[x+y*width + 1120 + 313*width] == color(120) || pixels[x+y*width + 1120 + 313*width] == color(150) || pixels[x+y*width + 1120 + 313*width] == color(105) || pixels[x+y*width + 1120 + 313*width] == color(165) || pixels[x+y*width + 1120 + 313*width] == color(195) || pixels[x+y*width + 1120 + 313*width] == color(255))
              amount[0] ++;
          }
        }
      
      
       
        if (pixels[313*width + x+y*width + 560] == color(205) || pixels[313*width + x+y*width + 560] == color(255)) {
          if (pixels[313*width + x+y*width] == color(225) || pixels[313*width + x+y*width] == color(195) || pixels[313*width + x+y*width] == color(210) || pixels[313*width + x+y*width] == color(90) || pixels[313*width + x+y*width] == color(165) || pixels[313*width + x+y*width] == color(180) || pixels[313*width + x+y*width] == color(90) || pixels[313*width + x+y*width] == color(120)  || pixels[313*width + x+y*width] == color(105)) {
            if (pixels[x+y*width + 1120 + 313*width] == color(135) || pixels[x+y*width + 1120 + 313*width] == color(105) || pixels[x+y*width + 1120 + 313*width] == color(120) || pixels[x+y*width + 1120 + 313*width] == color(195) || pixels[x+y*width + 1120 + 313*width] == color(165) ||  pixels[x+y*width + 1120 + 313*width] == color(150))
              amount[2] ++;
          }
        }
      
      
        
        if (pixels[313*width + x+y*width + 560] == color(205) || pixels[313*width + x+y*width + 560] == color(255)) {
          if (pixels[313*width + x+y*width] == color(210) || pixels[313*width + x+y*width] == color(150) || pixels[313*width + x+y*width] == color(165) || pixels[313*width + x+y*width] == color(90) || pixels[313*width + x+y*width] == color(120)) {
            if (pixels[x+y*width + 1120 + 313*width] == color(195) || pixels[x+y*width + 1120 + 313*width] == color(135) || pixels[x+y*width + 1120 + 313*width] == color(165) || pixels[x+y*width + 1120 + 313*width] == color(255) || pixels[x+y*width + 1120 + 313*width] == color(180))
              amount[5] ++;
          }
        }
      
      
     
        if (pixels[313*width + x+y*width] == color(165) || pixels[313*width + x+y*width] == color(120) || pixels[313*width + x+y*width] == color(105) || pixels[313*width + x+y*width] == color(90) || pixels[313*width + x+y*width] == color(210))
          amount[7] ++;
      
        
        if (pixels[313*width + x+y*width + 560] == color(205) || pixels[313*width + x+y*width + 560] == color(255)) {
          if (pixels[313*width + x+y*width] == color(180) || pixels[313*width + x+y*width] == color(165) || pixels[313*width + x+y*width] == color(90)) {
            if (pixels[x+y*width + 1120 + 313*width] == color(150) || pixels[x+y*width + 1120 + 313*width] == color(135))
              amount[6] ++;
          }
        }
      
      
        
        if (pixels[313*width + x+y*width + 560] == color(205) || pixels[313*width + x+y*width + 560] == color(255) || pixels[313*width + x+y*width + 560] == color(155)) {
          if (pixels[313*width + x+y*width] == color(210) || pixels[313*width + x+y*width] == color(180) || pixels[313*width + x+y*width] == color(165) || pixels[313*width + x+y*width] == color(150) || pixels[313*width + x+y*width] == color(120) || pixels[313*width + x+y*width] == color(105) || pixels[313*width + x+y*width] == color(90)) {
            if (pixels[x+y*width + 1120 + 313*width] == color(120) || pixels[x+y*width + 1120 + 313*width] == color(195) || pixels[x+y*width + 1120 + 313*width] == color(135) || pixels[x+y*width + 1120 + 313*width] == color(165) || pixels[x+y*width + 1120 + 313*width] == color(255) || pixels[x+y*width + 1120 + 313*width] == color(180))
              amount[8] ++;
          }
        }
      
      
       
        if (pixels[313*width + x+y*width + 560] == color(205) || pixels[313*width + x+y*width + 560] == color(255)) {
          if (pixels[313*width + x+y*width] == color(225) || pixels[313*width + x+y*width] == color(210) || pixels[313*width + x+y*width] == color(240) || pixels[313*width + x+y*width] == color(225) || pixels[313*width + x+y*width] == color(150) || pixels[313*width + x+y*width] == color(225)
               || pixels[313*width + x+y*width] == color(120) || pixels[313*width + x+y*width] == color(105) || pixels[313*width + x+y*width] == color(90)) {
            if (pixels[x+y*width + 1120 + 313*width] == color(135) || pixels[x+y*width + 1120 + 313*width] == color(120) || pixels[x+y*width + 1120 + 313*width] == color(165) || pixels[x+y*width + 1120 + 313*width] == color(255)
                || pixels[x+y*width + 1120 + 313*width] == color(180)) amount[9] ++;
                
          }
        }
      
      
        //bananas
        if (pixels[313*width + x+y*width + 560] == color(205)) {
          if (pixels[313*width + x+y*width] == color(180) || pixels[313*width + x+y*width] == color(165)) {
            if (pixels[x+y*width + 1120 + 313*width] == color(120) || pixels[x+y*width + 1120 + 313*width] == color(255))
              amount[10] ++;
          }
        }
      }
    }
    
    noStroke();
    textFont(font);
    textSize(10);
    textAlign(LEFT, UP);
    fill(255);
    text("RICE: " + amount[0], 590, 730);
    text("CORN: " + amount[1], 690, 730);
    text("SOYBEAN: " + amount[2], 790, 730);
    text("WHEAT: " + amount[3], 890, 730);
    text("POTATO: " + amount[4], 990, 730);
    text("TOMATO: " + amount[5], 1090, 730);
    text("SUGARCANE: " + amount[6], 1190, 730);
    text("CATTLE: " + amount[7], 590, 745);
    text("VEGETABLE: " + amount[8], 690, 745);
    text("APPLE: " + amount[9], 790, 745);
    text("BANANA: " + amount[10], 890, 745);
    text("MANGO: " + amount[11], 990, 745);
    text("SHEEP: " + amount[12], 1090, 745);
    text("ONION: " + amount[13], 1190, 745);
    text("BEAN: " + amount[14], 590, 760);
    text("PEANUT: " + amount[15], 690, 760);
    text("OLIVE: " + amount[16], 790, 760);
    text("RAPESEED: " + amount[17], 890, 760);
    text("ALMOND: " + amount[18], 990, 760);
    text("WALNUT: " + amount[19], 1090, 760);
  }
  
  if (keyPressed && key == CODED) { 
    if (keyCode == LEFT) {
      zoom[0] = 0;
      zoom[1] = 0;
    }
  }
}
