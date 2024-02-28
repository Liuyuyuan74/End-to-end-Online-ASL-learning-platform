import * as React from 'react';
import { useState } from 'react';
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

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    if (file) {
      const apiEndpoint = "https://mmvrj6fof8.execute-api.us-east-2.amazonaws.com/dev";
      const fileName = file.name;

      // Step 1: Request a pre-signed URL from your API
      fetch(apiEndpoint, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ imageName: fileName }),
      })
      .then(response => response.json())
      .then(data => {
        const preSignedUrl = data.url;
        console.log(preSignedUrl)
        
        // Step 2: Use the pre-signed URL to upload the file directly to S3
      //   return fetch(preSignedUrl, {
      //     method: 'PUT',
      //     body: file, // Upload the file directly to S3
      //     headers: {
      //       'Content-Type': 'image/png' // Adjust based on file type
      //     },
      //   });
      })
      // .then(uploadResponse => {
      //   if (uploadResponse.ok) {
      //     alert('Upload successful');
      //     const url = URL.createObjectURL(file);
      //     setFileInfo({ name: file.name, url: url });
      //   } else {
      //     alert('Upload failed');
      //   }
      // })
      .catch(error => {
        console.error('Error during file upload:', error);
      });
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
    </div>
  );
}
