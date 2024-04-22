import React, { useState } from 'react';
import { styled } from '@mui/material/styles';
import Button from '@mui/material/Button';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';

const VisuallyHiddenInput = styled('input')({
  clip: 'rect(0 0 0 0)',
  clipPath: 'inset(50%)',
  height: 1,
  overflow: 'hidden',
  position: 'absolute',
  whiteSpace: 'nowrap',
  width: 1,
});

export default function InputFileUpload() {
  const [fileInfo, setFileInfo] = useState({ name: "", url: "" });
  const [uploadResult, setUploadResult] = useState("");

  const handleFileChange = async (event) => {
    const file = event.target.files[0];
    if (file) {
      setFileInfo({ name: "", url: "" });
      setUploadResult("");
      const apiEndpoint = "/cgi-bin/upload.py"; 
      const formData = new FormData();
      formData.append('photo', file); 
      const url = URL.createObjectURL(file);
      try {
        const response = await fetch(apiEndpoint, {
          method: 'POST',
          body: formData, 
        });
        if (response.ok) {
          const data1 = await response.json();
          if(data1.SuccessCode === 2){
            alert(`Runtime error: ${data1.ErrorMessage}`);  
          }
          const data = JSON.parse(data1);
          console.log("Data received:", data); // Log the received data
          if(data.SuccessCode === 0){
            setFileInfo({ name: file.name, url: url });
            alert('Upload successful');
            setUploadResult(`This gesture means: ${data.InferResult}`); // Set uploadResult to the inferred result
          }else if(data.SuccessCode === 1){
            alert('Can\'t detect the landmark, please try other images');
            setUploadResult("Can\'t detect the landmark, please try other images");
          }
        } else {
          alert('Upload failed');
          setUploadResult("Upload failed. Please try again.");
        }
      } catch (error) {
        // console.error('Error during file upload:', error);
        setUploadResult("An error occurred during upload.");
      }
    }
  };

  return (
    <div>
      <Button component="label" variant="contained" startIcon={<CloudUploadIcon />}>
        Upload file
        <VisuallyHiddenInput type="file" onChange={handleFileChange} />
      </Button>
      {fileInfo.name && (
        <div>
          <p>File uploaded: {fileInfo.name}</p>
          {fileInfo.url && <img src={fileInfo.url} alt="Uploaded" style={{ maxWidth: '100%', height: 'auto' }} />}
        </div>
      )}
      <div style={{ fontSize: '24px', fontWeight: 'bold' }}>{uploadResult}</div>
    </div>
  );
}
