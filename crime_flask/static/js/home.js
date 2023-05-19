

let homebtnE=document.getElementById("homebtn");
let predictbtnE=document.getElementById("predictbtn");

let homeContainerE=document.getElementById("homecontainer");
let predictContainerE=document.getElementById("predictcontainer");



homebtnE.onclick = function(){
    predictContainerE.classList.add("d-none");
    homeContainerE.classList.remove("d-none");
};


predictbtnE.onclick = function(){
    predictContainerE.classList.remove("d-none");
    homeContainerE.classList.add("d-none");
};

