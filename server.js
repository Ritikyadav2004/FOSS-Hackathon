const express = require('express')
const bodyParser=require('body-parser')
const bcrypt=require('bcryptjs')
const mongoose=require('mongoose')
const app=express();
app.use(bodyParser.json())

mongoose.connect('mongodb://127.0.0.1:27017/FeedbackDb')
.then(()=>console.log("DbConnected"))
.catch((err)=>console.log('Error Occured-->',err))

app.get('/',(req,res)=>
{
    res.send("Server IS runnig at 3000")
});

//CREATING RAGISTER API
app.post('/register', async(req,res)=>
{
    const {username,password}=req.body;
    let users=[];
    // console.log("Ragistering succesfully");
    // console.log(username,password);

    // Check if user exists
    const exisitingUser=users.find(u=>u.username===username);
    if(exisitingUser) return res.status(404).json({message:"user already exist"})

    //hash the password
    const  hasedPassword=await bcrypt.hash(password,10);


    // save the password
    users.push({username,password:hasedPassword})
    res.json({message:"User registered successfully"})
});

app.listen(3000,()=>
{
    console.log("Srever Started runnin at port 3000");
}) 

