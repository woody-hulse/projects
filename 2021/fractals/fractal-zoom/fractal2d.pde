double scale = 4;
double translationX = -0.5;
double translationY = 0;
int resolution = 1000;
int maxDepth = 120;
int depthIncrease = int(maxDepth / 2);

void setup() {
  
  size(1000, 1000);
  background(255);  
  
   // font = loadFont("Monospaced-20.vlw");
  displayFractal(translationX, translationY, scale, resolution, maxDepth);
}

Complex fractal(Complex z, Complex c) {
  return z.m(z).a(c);
}

Complex findValue(Complex z, Complex c, int depth, int maxDepth) {
  
  if (depth == maxDepth) {
    return z;
  }
  
  if (z.complexMagnitude() > 100){
    Complex x = new Complex(0, 0);
    x.depth = depth;
    return x;
  }
  
  return findValue(fractal(z, c), c, depth + 1, maxDepth);
}

Complex[][] calculateFractal(double translationX, double translationY, double scale, int resolution, int maxDepth) {
  
  Complex[][] image = new Complex[resolution][resolution];
  
  for (int i = 0; i < resolution; i++) {
    for (int j = 0; j < resolution; j++) {
      Complex point = new Complex((i / (double)resolution - 0.5d) * scale + translationX, (j / (double)resolution - 0.5d) * scale + translationY);
      
      image[i][j] = findValue(new Complex(0, 0), point, 0, maxDepth);
    }
  }
  
  return image;
}

color calculateFill(Complex c) {
  
  double value = 0;
  if (c.a == 0d && c.b == 0d) {
    value = c.depth;
  }
  
  value *= 255d / maxDepth * 2;
  
  colorMode(HSB);
  if (value == 0) return color(0, 0, 0);
  else return color((200 - (float)value) % 255, 255, 255);
}

void displayFractal(double translationX, double translationY, double scale, int resolution, int maxDepth) {
  
  background(0);
  
  Complex[][] image = calculateFractal(translationX, translationY, scale, resolution, maxDepth);
  
  noStroke();
  float size = width / resolution;
  for (int i = 0; i < resolution; i++) {
    for (int j = 0; j < resolution; j++) {
      fill(calculateFill(image[i][j]));
      rect(i * size, j * size, size, size);
    }
  }
  
  textAlign(RIGHT, DOWN);
  textSize(5);
  fill(240);
  text("scale: " + scale, width - 50, height - 50);
  
}

void mouseClicked() {
  
  translationX += (mouseX - width / 2) * scale / width;
  translationY += (mouseY - height / 2) * scale / height;
  
  if (mouseButton == LEFT) {
    scale *= 0.5;
    maxDepth += depthIncrease;
  } else if (mouseButton == RIGHT) {
    scale *= 2;
    maxDepth -= depthIncrease;
  }
  
  displayFractal(translationX, translationY, scale, resolution, maxDepth);
  
}

void draw() {}
