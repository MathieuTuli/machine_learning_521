var droop = [];
var globalCounter = 0;
var mSymbol;
// var orange = rgb(255, 176, 0);
// var blue = rgb(19, 202, 251);

function setup(){
  createCanvas(windowWidth,windowHeight);
  background(19, 202, 251);

  var tempDroop = [];
  mSymbol = new M(50,windowWidth/4,3*windowWidth/4,windowHeight)

  for(i=0;i<width;i+=width/40){
    // if(random(1,5)>4){
    //   var tempWeight = random(40,60);
    //   var tempRate = random(0.1,2);
    //   tempDroop.push(new SingleDroop(tempWeight,i,0,"orange",tempRate));
    //   tempDroop.push(new SingleDroop(tempWeight,i,0,"blue",tempRate));
    // }
    // else{
    //   tempDroop.push(new SingleDroop(random(40,100),i,0,"orange",random(0.1,2)));
    // }
    tempDroop.push(new SingleDroop(random(40,100),i,0,"orange",random(3,7)));

  }
  droop = tempDroop;
}
function draw(){
  for(i=0;i<droop.length;i++){
    droop[i].draw_droop();
    droop[i].y += droop[i].rate;
  }
  globalCounter++;
  if(globalCounter>150){
    mSymbol.draw_M();
  }

}
class M {
  constructor(weight, lVert, rVert, height) {
    this.weight = weight;
    this.lVert = lVert;
    this.rVert= rVert;
    this.height = height;
  }
  draw_M(){
    stroke(255, 176, 0);
    strokeWeight(this.weight);
    line(this.lVert,this.height,this.lVert,0);
    line(this.rVert,this.height,this.rVert,0);
    line(this.lVert,0,(this.rVert-this.lVert),windowHeight);
    line(this.rVert,0,(this.rVert-this.lVert),windowHeight);
  }
}
class SingleDroop{
  constructor(weight,x,y,colour,rate){
    this.weight = weight;
    this.x = x;
    this.y = y;
    this.colour = colour;
    this.rate = rate;
  }
  draw_droop(){
    if(this.colour == "orange"){
      if(globalCounter>150){
        stroke(19, 202, 251);
      }
      else{
        stroke(255, 176, 0);
      }
      strokeWeight(this.weight);
      line(this.x,0,this.x,this.y);
      ellipse(this.x,this.y,1);
    }
    else{
      stroke(19, 202, 251);
      // stroke(0,0,0);
      strokeWeight(this.weight);
      line(this.x,windowHeight,this.x,this.y);
      ellipse(this.x,this.y,1);
    }
  }
}
