function openModal1(){
  document.getElementById('modal-1').style.display = 'block';
  $('body').addClass('modal-is-open');
}

function closeModal1(){
  document.getElementById('modal-1').style.display = 'none';
  $('body').removeClass('modal-is-open');
}
function openModal2(){
  document.getElementById('modal-2').style.display = 'block';
  $('body').addClass('modal-is-open');
}

function closeModal2(){
  document.getElementById('modal-2').style.display = 'none';
  $('body').removeClass('modal-is-open');
}
function openModal3(){
  document.getElementById('modal-3').style.display = 'block';
  $('body').addClass('modal-is-open');
}

function closeModal3(){
  document.getElementById('modal-3').style.display = 'none';
  $('body').removeClass('modal-is-open');
}
function openModal4(){
  document.getElementById('modal-4').style.display = 'block';
  $('body').addClass('modal-is-open');
}

function closeModal4(){
  document.getElementById('modal-4').style.display = 'none';
  $('body').removeClass('modal-is-open');
}
function openModal5(){
  document.getElementById('modal-5').style.display = 'block';
  $('body').addClass('modal-is-open');
}

function closeModal5(){
  document.getElementById('modal-5').style.display = 'none';
  $('body').removeClass('modal-is-open');
}
function openModal6(){
  document.getElementById('modal-6').style.display = 'block';
  $('body').addClass('modal-is-open');
}

function closeModal6(){
  document.getElementById('modal-6').style.display = 'none';
  $('body').removeClass('modal-is-open');
}
function openModal7(){
  document.getElementById('modal-7').style.display = 'block';
  $('body').addClass('modal-is-open');
}

function closeModal7(){
  document.getElementById('modal-7').style.display = 'none';
  $('body').removeClass('modal-is-open');
}
function openModal8(){
  document.getElementById('modal-8').style.display = 'block';
  $('body').addClass('modal-is-open');
}

function closeModal8(){
  document.getElementById('modal-8').style.display = 'none';
  $('body').removeClass('modal-is-open');
}
function openModal9(){
  document.getElementById('modal-9').style.display = 'block';
  $('body').addClass('modal-is-open');
}

function closeModal9(){
  document.getElementById('modal-9').style.display = 'none';
  $('body').removeClass('modal-is-open');
}
function openModal10(){
  document.getElementById('modal-10').style.display = 'block';
  $('body').addClass('modal-is-open');
}

function closeModal10(){
  document.getElementById('modal-10').style.display = 'none';
  $('body').removeClass('modal-is-open');
}
function openModal11(){
  document.getElementById('modal-11').style.display = 'block';
  $('body').addClass('modal-is-open');
}

function closeModal11(){
  document.getElementById('modal-11').style.display = 'none';
  $('body').removeClass('modal-is-open');
}
function openModal12(){
  document.getElementById('modal-12').style.display = 'block';
  $('body').addClass('modal-is-open');
}

function closeModal12(){
  document.getElementById('modal-12').style.display = 'none';
  $('body').removeClass('modal-is-open');
}
function openModal13(){
  document.getElementById('modal-13').style.display = 'block';
  $('body').addClass('modal-is-open');
}

function closeModal13(){
  document.getElementById('modal-13').style.display = 'none';
  $('body').removeClass('modal-is-open');
}
function openModal14(){
  document.getElementById('modal-14').style.display = 'block';
  $('body').addClass('modal-is-open');
}

function closeModal14(){
  document.getElementById('modal-14').style.display = 'none';
  $('body').removeClass('modal-is-open');
}
function openModal15(){
  document.getElementById('modal-15').style.display = 'block';
  $('body').addClass('modal-is-open');
}

function closeModal15(){
  document.getElementById('modal-15').style.display = 'none';
  $('body').removeClass('modal-is-open');
}
window.onclick = function(event){
  if(event.target == document.getElementById('modal-1')){
    document.getElementById('modal-1').style.display = 'none';
    $('body').removeClass('modal-is-open');
  }
  else if(event.target == document.getElementById('modal-2')){
    document.getElementById('modal-2').style.display = 'none';
    $('body').removeClass('modal-is-open');
  }
  else if(event.target == document.getElementById('modal-3')){
    document.getElementById('modal-3').style.display = 'none';
    $('body').removeClass('modal-is-open');
  }
  else if(event.target == document.getElementById('modal-4')){
    document.getElementById('modal-4').style.display = 'none';
    $('body').removeClass('modal-is-open');
  }
  else if(event.target == document.getElementById('modal-5')){
    document.getElementById('modal-5').style.display = 'none';
    $('body').removeClass('modal-is-open');
  }
  else if(event.target == document.getElementById('modal-6')){
    document.getElementById('modal-6').style.display = 'none';
    $('body').removeClass('modal-is-open');
  }
  else if(event.target == document.getElementById('modal-7')){
    document.getElementById('modal-7').style.display = 'none';
    $('body').removeClass('modal-is-open');
  }
  else if(event.target == document.getElementById('modal-8')){
    document.getElementById('modal-8').style.display = 'none';
    $('body').removeClass('modal-is-open');
  }
  else if(event.target == document.getElementById('modal-9')){
    document.getElementById('modal-9').style.display = 'none';
    $('body').removeClass('modal-is-open');
  }
  else if(event.target == document.getElementById('modal-10')){
    document.getElementById('modal-10').style.display = 'none';
    $('body').removeClass('modal-is-open');
  }
  else if(event.target == document.getElementById('modal-11')){
    document.getElementById('modal-11').style.display = 'none';
    $('body').removeClass('modal-is-open');
  }
  else if(event.target == document.getElementById('modal-12')){
    document.getElementById('modal-12').style.display = 'none';
    $('body').removeClass('modal-is-open');
  }
  else if(event.target == document.getElementById('modal-13')){
    document.getElementById('modal-13').style.display = 'none';
    $('body').removeClass('modal-is-open');
  }
  else if(event.target == document.getElementById('modal-14')){
    document.getElementById('modal-14').style.display = 'none';
    $('body').removeClass('modal-is-open');
  }
  else if(event.target == document.getElementById('modal-15')){
    document.getElementById('modal-15').style.display = 'none';
    $('body').removeClass('modal-is-open');
  }
};
var counter = 0;
var ccounter = -500;

function rrandom(max, min, offset){
  return (Math.random() * (max-min) + min + offset);
}

function hasResized(){
  // var greetingTop =  $('#greetingText').offset().top;
  // var greetingHeight = $('#greetingText').outerHeight();
  // var greetingLeft =  $('#greetingText').offset().left;
  //
  // var educationTop = $('#educationText').offset().top + 80;
  // var educationLeft = $('#educationText').offset().left;
  // var educationWidth = $('#educationText').width();
  // var educationHeight = $('#educationText').height();
  //
  // var workTop = $('#workID').offset().top + 80;
  // var workHeight = $('#workHeight').height();
  // var workLeft;
  //
  // var projectTop = $('#projectID').offset().top + 80;
  // var projectLeft;
  // var porjectHeight = $('#projectHeight').height();
  //
  // var contactTop = $('#contactID').offset().top + 80;
  // var contactLeft;
  // var contactHeight = $('#contactHeight').height();
  //
  // var educationAngle = Math.atan(educationWidth/((greetingHeight+greetingTop)-educationTop));
  // var workAngle;
  // var projectAngle;
  // var contactAngle;

  var docWidth = $(window).width();
  var docHeight = $(window).height();

  // document.getElementById('main').style.width = docWidth + "px";
  // document.getElementById('main').style.height = docHeight + "px";

  // document.getElementById('educationBackground').style.marginTop = educationTop + "px";
  // // document.getElementById('educationBackground').style.transform = "rotate(" + educationAngle + "deg)";
  // document.getElementById('educationBackground').style.transform = "rotate(60deg)";
  //
  //
  // document.getElementById('workBackground').style.marginTop = workTop*1.17   + "px";
  // document.getElementById('workBackground').style.transform = "rotate(105deg) translateY("+docWidth+"px)";
  //
  // document.getElementById('projectsBackground').style.marginTop = projectTop*0.94 + "px";
  // document.getElementById('projectsBackground').style.transform = "rotate(60deg)";
  //
  // document.getElementById('contactBackground').style.marginTop = contactTop*1.08 + "px";
  // document.getElementById('contactBackground').style.transform = "rotate(105deg) translateY("+docWidth+"px)";

  // document.getElementById('educationBackgroundOne').style.width = docWidth + "px";
  // document.getElementById('workBackgroundOne').style.width = docWidth + "px";
  // document.getElementById('projectsBackgroundOne').style.width = docWidth + "px";
  // document.getElementById('contactBackgroundOne').style.width = docWidth + "px";
  //
  // document.getElementById('educationBackgroundTwo').style.borderWidth = "0 0 " + docWidth/2.66 + "px " + docWidth + "px";
  // document.getElementById('workBackgroundTwo').style.borderWidth = "0 0 " + docWidth/2.66 + "px " + docWidth + "px";
  // document.getElementById('projectsBackgroundTwo').style.borderWidth = "0 0 " + docWidth/2.66 + "px " + docWidth + "px";
  // document.getElementById('contactBackgroundTwo').style.borderWidth = "0 0 " + docWidth/2.66 + "px " + docWidth + "px";
  //
  // var random1 = 0;
  // var random2 = 0;
  // var random3 = 0;
  // var random4 = 0;
  // random1 = rrandom(docHeight/4,0,docWidth/2.66);
  //
  // document.getElementById('educationBackgroundTwo').style.height = random1 + "px";
  // document.getElementById('educationBackgroundThree').style.height = random2 + docWidth/2.66 + "px";
  //
  // random2 = rrandom(docHeight/2,docHeight/4,docWidth/2.66);
  //
  // document.getElementById('workBackgroundOne').style.marginTop = random2 + "px";
  // document.getElementById('workBackgroundThree').style.height =  random3 + docWidth/3.66 + "px";
  //
  // random3 = rrandom(docHeight/3,docHeight/4,docWidth/2.66);
  //
  // document.getElementById('projectsBackgroundOne').style.marginTop = random3+ "px";
  // document.getElementById('projectsBackgroundThree').style.height =   random4 + docWidth/3 + "px";
  //
  // random4 = rrandom(docHeight,docHeight/3,docWidth/2.66);
  //
  // document.getElementById('contactBackgroundOne').style.marginTop = random4 + "px";
  // document.getElementById('contactBackgroundThree').style.height =  0 + "px";
  // document.getElementById('educationBackgroundTwo').style.height = educationTop + "px";
  // document.getElementById('educationBackgroundThree').style.height = workHeight + educationHeight + (educationTop-(greetingHeight)) + "px";
  //
  // document.getElementById('workBackgroundOne').style.marginTop = workTop + "px";
  // document.getElementById('workBackgroundThree').style.height =  workHeight*1.3 + "px";
  //
  // document.getElementById('projectsBackgroundOne').style.marginTop = projectTop + "px";
  // document.getElementById('projectsBackgroundThree').style.height =  workHeight + "px";
  //
  // document.getElementById('contactBackgroundOne').style.marginTop = contactTop-contactHeight + "px";
  // document.getElementById('contactBackgroundThree').style.height =  workHeight + "px";

}
var didResize = true;

$(window).resize(function (event) {
  didResize = true;
});

setInterval(function () {
  if (didResize) {
    // hasResized();
    didResize = false;
  }
},250);
// hasResized();

$(document).ready(function(){



  navHeight = $('nav').outerHeight();
  document.getElementById('footer').style.marginTop = navHeight + "px";
  // document.getElementById('iceberg').style.backgroundSize = window.outerHeight + "px " + window.outerWidth + "px";
  // document.getElementById('icebergTop').style.height = window.outerHeight + "px " + window.outerWidth + "px";
  // document.getElementById('main').style.height = window.outerHeight + "px";
  // document.getElementById('body').style.height = window.outerHeight + "px";
  // document.getElementById('video').style.width = $(window).width() +"px";
  // document.getElementById('video').style.height = $(window).height() +"px";
  // document.getElementById('main').style.height = "0px";
  //
  // setInterval(function () {
  //   document.getElementById('main').style.height = "auto";
  //   // document.getElementById('video').style.width = "0px";
  //   document.getElementById('video').style.height = "0px";
  //   // $.get('footer.html', function(data) {
  //   //   $("#footer-placeholder").replaceWith(data);
  //   // });
  // },1500);

  var skillsOffset = $("#skills").offset().top;
  var currentPosition;
  var didScroll = false;
  var didResize = true;

  $(window).scroll(function (event) {
    didScroll = true;
  });
  $(window).resize(function (event) {
    didResize = true;
  });

  // setInterval(function () {
  //   if (didScroll) {
  //     hasScrolled();
  //     didScroll = false;
  //   }
  //   if (didResize) {
  //     hasResized();
  //     didResize = false;
  //   }
  // },0);

  var carTimer = setInterval(function () {
    // driveCar();
  },0);

  function hasScrolled(){
    currentPosition = $(window).scrollTop() + $(window).height();
    if (currentPosition >= skillsOffset){
      document.getElementById('html').style.width = "75%";
      document.getElementById('CSS').style.width = "75%";
      document.getElementById('javascript').style.width = "75%";
      document.getElementById('C').style.width = "85%";
      document.getElementById('python').style.width = "75%";
      document.getElementById('AWS').style.width = "80%";
      document.getElementById('java').style.width = "70%";
      document.getElementById('verilog').style.width = "75%";
      document.getElementById('assembly').style.width = "75%";
      document.getElementById('soldering').style.width = "95%";
      document.getElementById('machine').style.width = "75%";
      document.getElementById('git').style.width = "100%";
    }
    else{
      document.getElementById('html').style.width = "0%";
      document.getElementById('CSS').style.width = "0%";
      document.getElementById('javascript').style.width = "0%";
      document.getElementById('C').style.width = "0%";
      document.getElementById('python').style.width = "0%";
      document.getElementById('AWS').style.width = "0%";
      document.getElementById('java').style.width = "0%";
      document.getElementById('verilog').style.width = "0%";
      document.getElementById('assembly').style.width = "0%";
      document.getElementById('soldering').style.width = "0%";
      document.getElementById('machine').style.width = "0%";
      document.getElementById('git').style.width = "0%";
    }
  }

  function hasResized(){
    // document.getElementById('body').style.maxHeight = "300px";//$('#footerID').offset().top + $('#footerID').height() + "px";

  }
});
