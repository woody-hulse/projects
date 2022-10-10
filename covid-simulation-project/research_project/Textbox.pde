class Textbox {
  
   PVector pos;
   PVector dim;
   float size = 20;
   int index;
   
   float disp = spacingLength/16;
   
   String text = "";

   boolean selected = true;
   
   Textbox(PVector pos, PVector dim, int index) {
      this.pos = pos;
      this.dim = dim;
      
      this.index = index;
      
      if (index != -1) {
        Classroom c = classrooms.get(index);
        text = c.name;
      }
   }
   
   void display() {
      
     fill(200, 20);
     noStroke();
     rect(pos.x + disp, pos.y + disp*2, textWidth(text) + disp*2, dim.y - disp*4, 10);
     strokeWeight(1);
      
     fill(foreground);
     textSize(size);
     textAlign(LEFT, CENTER);
     if (index == -1) 
       text(text, pos.x + (textWidth("a") / 2) + 1.5, pos.y + size + 7);
   }
   
   boolean keyPress(char k, int code) {
     
      if (selected) {
         if (code == (int)BACKSPACE && text.length() > 0) {
            text = text.substring(0, text.length() - 1);
         } else if (code == (int)ENTER) {
            return true;
         } else if (textWidth(text + k) < dim.x - 155) {
           text += k;
         }
      }
      
      classrooms.get(index).name = text;
      
      return false;
   }
   
   //is mouse over box
   boolean mouseOver() {
    
     if (mouseX >= pos.x && mouseX <= pos.x + dim.x) {
       if (mouseY >= pos.y && mouseY <= pos.y + dim.y) {
         return true;
       }
     }
      
      return false;
   }
}
