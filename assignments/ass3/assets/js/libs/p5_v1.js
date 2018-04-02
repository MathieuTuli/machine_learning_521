var width;
var lines = [];
var mLines = [];
var globalCounter = 0;
function setup() {
  frameRate(100000000);

  var lowerRandom = 10;
  var upperRandom = 14;
  var vertRate = 90;
  var diagRate = 30;

  if(windowHeight>windowWidth){
    width=windowHeight;
  }
  if(windowWidth>300){
    lowerRandom += 10;
    upperRandom += 10;
  }

  // createCanvas(windowWidth, windowHeight);
  createCanvas(windowWidth, windowHeight);
  background(245);

  var tempLines = [];
  var tempMLines = [];
  var counter = 0;
  var quart = true;
  var thirdQuart = true;

  var divisor = 40;

  tempMLines.push(new Line(windowWidth/4,windowHeight,windowWidth/4,0,"Bvert",vertRate));
  tempMLines.push(new Line(windowWidth/4,0,windowWidth/2,windowHeight-windowWidth/4,"TR",diagRate));
  tempMLines.push(new Line(3*windowWidth/4,windowHeight,3*windowWidth/4,0,"Bvert",vertRate));
  tempMLines.push(new Line(3*windowWidth/4,0,windowWidth/2,3*windowWidth/4,"TL",diagRate));

  // tempMLines.push(new Line(windowWidth/2,windowHeight,windowWidth/2,150,"Bvert",diagRate));
  // tempMLines.push(new Line(windowWidth/4,150,3*windowWidth/4,150,"hor",diagRate));
  // tempMLines.push(new Line(3*windowWidth/4,150,windowWidth/4,150,"Rhor",diagRate));

  for (i=0;i<width;i+=(width/40)){
    // if(i >= width/4 && quart){
    //   console.log("yed")
    //   tempMLines.push(new Line(i,width,i,-2,"Bvert",14));
    //   tempMLines.push(new Line(i,0,width/2+2,width-i,"TR",9));
    //   quart = false;
    // }
    // if( i >= width*3/4 && thirdQuart){
    //   console.log("yo")
    //   tempMLines.push(new Line(i,width,i,-2,"Bvert",14));
    //   tempMLines.push(new Line(i,0,width/2-2,i,"TL",9));
    //   thirdQuart = false;
    // }

    if(counter%2 == 0){
      tempLines.push(new Line(0,width-i,i,width,"TR",random(lowerRandom,upperRandom)));
      tempLines.push(new Line(i,0,width,width-i,"TR",random(lowerRandom,upperRandom)));

      tempLines.push(new Line(i,width,width,i,"BR",random(lowerRandom,upperRandom)));
      tempLines.push(new Line(0,i,i,0,"BR",random(lowerRandom,upperRandom)));

      tempLines.push(new Line(i,0,i,width,"vert",random(lowerRandom,upperRandom)));
      tempLines.push(new Line(0,i,width,i,"hor",random(lowerRandom,upperRandom)));
    }
    else{
      tempLines.push(new Line(i,width,0,width-i,"BL",random(lowerRandom,upperRandom)));
      tempLines.push(new Line(width,width-i,i,0,"BL",random(lowerRandom,upperRandom)));

      tempLines.push(new Line(i,0,0,i,"TL",random(lowerRandom,upperRandom)));
      tempLines.push(new Line(width,i,i,width,"TL",random(lowerRandom,upperRandom)));

      tempLines.push(new Line(i,width,i,0,"Bvert",random(lowerRandom,upperRandom)));
      tempLines.push(new Line(width,i,0,i,"Rhor",random(lowerRandom,upperRandom)));
    }
    // tempLines.push(new Line(i,width,0,width-i,"BL",random(lowerRandom,upperRandom)));
    // tempLines.push(new Line(0,width-i,i,width,"TR",random(lowerRandom,upperRandom)));
    // tempLines.push(new Line(i,0,width,width-i,"TR",random(lowerRandom,upperRandom)));

    // tempLines.push(new Line(i,width,width,i,"BR",random(lowerRandom,upperRandom)));
    // tempLines.push(new Line(0,i,i,0,"BR",random(lowerRandom,upperRandom)));
    // tempLines.push(new Line(i,0,0,i,"TL",random(lowerRandom,upperRandom)));

    // tempLines.push(new Line(i,0,i,width,"vert",random(lowerRandom,upperRandom)));
    // tempLines.push(new Line(0,i,width,i,"hor",random(lowerRandom,upperRandom)));
    counter++;
  }

  mLines = tempMLines;
  lines = tempLines;
}

function draw(){
  if(globalCounter <= windowWidth/20){
    for(i=0;i<lines.length;i++){
      stroke(0,0,0);
      strokeWeight(3);
      secondaryDraw(lines[i]);
    }
  }

  if(globalCounter > windowWidth/20){
    for(i=0;i<mLines.length;i++){
      stroke(0,0,0);
      strokeWeight(10);
      secondaryDraw(mLines[i]);
    }
  }
  globalCounter++;
}

function secondaryDraw(lines){
  if(lines.direction == "TR"){
    if(lines.tempX <= lines.endX){
      lines.draw_line();
      lines.tempX += 1*lines.rate;
      lines.tempY += 1*lines.rate;
    }
  }
  else if(lines.direction == "BL"){
    if(lines.tempX >= lines.endX){
      lines.draw_line();
      lines.tempX -= 1*lines.rate;
      lines.tempY -= 1*lines.rate;
    }
  }
  else if(lines.direction == "BR"){
    if(lines.tempX <= lines.endX){
      lines.draw_line();
      lines.tempX += 1*lines.rate;
      lines.tempY -= 1*lines.rate;
    }
  }
  else if(lines.direction == "TL"){
    if(lines.tempX >= lines.endX){
      lines.draw_line();
      lines.tempX -= 1*lines.rate;
      lines.tempY += 1*lines.rate;
    }
  }
  else if(lines.direction == "vert"){
    if(lines.tempY <= lines.endY){
      lines.draw_line();
      lines.tempY += 1*lines.rate;
    }
  }
  else if(lines.direction == "hor"){
    if(lines.tempX <= lines.endX){
      lines.draw_line();
      lines.tempX  += 1*lines.rate;
    }
  }
  else if(lines.direction == "Bvert"){
    if(lines.tempY >= lines.endY){
      lines.draw_line();
      lines.tempY -= 1*lines.rate;
    }
  }
  else if(lines.direction == "Rhor"){
    if(lines.tempX >= lines.endX){
      lines.draw_line();
      lines.tempX  -= 1*lines.rate;
    }
  }
}

class Line {
  constructor(startX,startY,endX,endY,direction,rate){
    this.startX = startX;
    this.startY = startY;
    this.endX = endX;
    this.endY = endY;
    this.tempX = startX;
    this.tempY = startY;
    this.direction = direction; //0 mens top left to bottom right, 1 means bottom right to top left, 2 vert, 3 hor
    this.rate = rate;
  }
  draw_line(){
    line(this.startX,this.startY,this.tempX,this.tempY);
  }
}
