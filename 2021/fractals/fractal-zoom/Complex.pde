class Complex {
    double a;
    double b;
    
    int depth;

    public Complex(double a, double b) {
        this.a = a;
        this.b = b;
        
        depth = 0;
    }
    
    public Complex a(Complex c) {
      double a = this.a + c.a;
      double b = this.b + c.b;
      return new Complex(a, b);
    }
    
    public Complex s(Complex c) {
      double a = this.a - c.a;
      double b = this.b - c.b;
      return new Complex(a, b);
    }

    public Complex m(Complex c) {
      double a = this.a * c.a - this.b * c.b;
      double b = this.a * c.b + this.b * c.a;
      return new Complex(a, b);
    }
    
    public Complex d(Complex c) {
      double a = (this.a * c.a + this.b * c.b) / (c.a * c.a + c.b * c.b);
      double b = (this.b * c.a - this.a * c.b) / (c.a * c.a + c.b * c.b);
      return new Complex(a, b);
    }
    
    public double complexMagnitude() {
      return sqrt((float)(this.a * this.a + this.b * this.b));
    }
}
