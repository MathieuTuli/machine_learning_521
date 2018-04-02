//place the navbar
var navbarOffset;
var navbarHeight;
var navInitialHeight;
$(document).ready(function(){
  //Globals:
  var didScroll;
  navbarOffset = 0;//$('.logos').outerHeight();
  navbarHeight = $('nav').outerHeight();
  var navbarInitialHeight = $('nav').outerHeight();
  console.log(navbarInitialHeight);
  // document.getElementById('main').style.marginTop = navbarInitialHeight + "px";
  scrollHide = 200;

  $(window).scroll(function (event) {
    didScroll = true;
  });

  // check scroll at interval: run hasScroll
  // setInterval(function () {
  //   if (didScroll) {
  //     hasScrolled();
  //     didScroll = false;
  //   }
  // },50);

  var prevScrollPos = 0;
  function hasScrolled() {
    var currentScrollPos = $(this).scrollTop();
    if(currentScrollPos>=scrollHide && currentScrollPos >= prevScrollPos){
      //$('nav').removeClass('nav-fix').addClass('nav-unfixed');
      document.getElementById('fixableNav').style.top = -navbarHeight-10 +"px";
      // $('main').removeClass('main').addClass('main-offset');
      //document.getElementById('main').style.paddingTop = navbarHeight + "px";
    }
    else{
      if(currentScrollPos + $(window).height() < $(document).height()) {
        document.getElementById('fixableNav').style.top = "0px";
        //$('nav').removeClass('nav-unfixed').addClass('nav-fix');
        // $('main').removeClass('main-offset').addClass('main');
        //document.getElementById('main').style.paddingTop =  "0px";
        }
    }
    prevScrollPos = currentScrollPos;
  }
});
function openNavbar() {
    if (document.getElementById("the-navbar").className === "navbar-links") {
        document.getElementById("the-navbar").className += " responsive";
        navbarHeight=$('nav').outerHeight();
        if( $(this).scrollTop()>=navbarOffset){
          document.getElementById('main').style.paddingTop = navbarHeight + "px";
        }
    }
    else {
        document.getElementById("the-navbar").className = "navbar-links";
        navbarHeight=navbarInitialHeight;
        if( $(this).scrollTop()>=navbarOffset){
          document.getElementById('main').style.paddingTop = navbarInitialHeight + "px";
        }
        else{
          document.getElementById('main').style.paddingTop = "0px";
        }
    }
}
