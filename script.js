let video;
let canvas;
let nameInput;

function init(){
    video=document.getElementById('video')
    canvas=document.getElementById('canvas')
    nameInput=document.getElementById('name')

    navigator.mediaDevices.getUserMedia({video: true})
    .then(stream=>{
        video.srcObject=stream
    })
    .catch(error=>{
        console.log("error access webcam",error)
        alert("cannot acccess webcam")
    });
}

function capture(){
    const context=canvas.getContext('2d')
    context.drawImage(video,0,0,canvas.width,canvas.height)
    canvas.style.display='block'
    video.style.display='none'
}

function register(){
    const name=nameInput.value;
    const photo=dataURItoBlob(canvas.toDataURL());

    if(!name || !photo){
        alert("name and photo are required");
        return;
    }
    const formdata=new FormData();
    formdata.append('name', name);
    formdata.append('photo',photo,"register.jpg");
    fetch("/register",{
        method:'POST',
        body:formdata
    })
    .then(response=>response.json())
    .then(data=>{
        if(data.success){
            alert("data success register")
            window.location.href='/';
        }
        else{
            alert("sorry failed register")
        }

    })
    .catch(error=>{
        console.log("error",error);
    });


}

function login(){
    const context=canvas.getContext('2d');
    context.drawImage(video,0,0,canvas.width,canvas.height);
    const photo=dataURItoBlob(canvas.toDataURL());

    if(!photo){
        alert("name and photo are required");
        return;
    }
    const formdata=new FormData();
    formdata.append('photo',photo,"login.jpg");
    fetch("/login",{
        method:'POST',
        body:formdata
    })
    .then(response=>response.json())
    .then(data=>{
        console.log(data);
        if(data.success){
            alert("login success")
            window.location.href='/success?user_name='+nameInput.value;
        }
        else{
            alert("sorry failed login")
        }
    }).catch(error=>{
        console.log("error",error);
    });

}

function dataURItoBlob(dataURI){
    const bytestring=atob(dataURI.split(',')[1]);
    const mimeString=dataURI.split(',')[0].split(':')[1].split(';')[0];
    const ab=new ArrayBuffer(bytestring.length);
    const ia=new Uint8Array(ab);
    for(let i=0;i<bytestring.length;i++){
        ia[i]=bytestring.charCodeAt(i);
    }
    return new Blob([ab],{type:mimeString});

}

init()
