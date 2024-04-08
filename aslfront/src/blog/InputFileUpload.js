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
      const apiEndpoint = "/cgi-bin/upload.py"; 
      const formData = new FormData();
      formData.append('photo', file); 
      const url = URL.createObjectURL(file);
      setFileInfo({ name: file.name, url: url });

      try {
        const response = await fetch(apiEndpoint, {
          method: 'POST',
          body: formData, 
        });

        if (response.ok) {
          // const data = await response.json(); // Adjust according to the actual response format
          alert('Upload successful');
          // console.log(data); // Log the response data
          // setUploadResult(data.result || "Upload successful. Please check your email for confirmation."); 
        } else {
          alert('Upload failed');
          setUploadResult("Upload failed. Please try again."); 
        }
      } catch (error) {
        console.error('Error during file upload:', error);
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
      <div>{uploadResult}</div>
    </div>
  );
}
