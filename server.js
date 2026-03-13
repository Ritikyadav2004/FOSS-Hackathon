const express = require('express')
const bodyParser=require('body-parser')
const bcrypt=require('bcryptjs')
const mongoose=require('mongoose')
const User = require('./models/user');
const { JsonWebTokenError } = require('jsonwebtoken');
const app = express();
const jwt=require('jsonwebtoken')


app.use(bodyParser.json())


mongoose.connect('mongodb://127.0.0.1:27017/FeedbackDb')
.then(()=>console.log("DbConnected"))
.catch((err)=>console.log('Error Occured-->',err))

app.get('/',(req,res)=>
{
    res.send("Server IS runnig at 3000")
});



//CREATING RAGISTER API
app.post('/register', async (req,res)=>{
    const {username,email,password} = req.body;

    // Check if user exists in MongoDB
    const exisitingUser = await User.findOne({ username: email});

    if(exisitingUser){
        return res.status(400).json({message:"User already exist"});
    }

    // hash password
    const hashedPassword = await bcrypt.hash(password,10);

    // create user
    const newUser = new User({
        email:email,
        username: username,
        password: hashedPassword
    });

    // save in MongoDB
    await newUser.save();

    res.status(200).json({message:"User registered successfully"});
});


   app.post('/login',async (req,res)=>
{
    const{email,password}=req.body;
    const user= await User.findOne({email});
    if(!user)
    {
        return res.status(400).json({
            message:"User Not Found"
        })
    }

    const isMatch=await bcrypt.compare(password,user.password)
    if(!isMatch)
    {
        return res.status(400).json({
            message:"Incorrect Password"
        })
    }

    const token = jwt.sign(
        {id:user._id},
        "JHBFIUWBFIUWB",
        {expiresIn:'1h'}
    )

    res.json({
        message:"Login Succesfully",
        token:token
    })
})

app.listen(3000,()=>
{
    console.log("Srever Started runnin at port 3000");
}) 

