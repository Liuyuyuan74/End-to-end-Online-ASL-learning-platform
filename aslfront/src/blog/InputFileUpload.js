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
  bottom: 0,
  left: 0,
  whiteSpace: 'nowrap',
  width: 1,
});

export default function InputFileUpload() {
  const [fileInfo, setFileInfo] = useState({ name: "", url: "" });

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    if (file) {
      const url = URL.createObjectURL(file);
      setFileInfo({ name: file.name, url: url });
    }
  };

  return (
    <div>
      <Button component="label" variant="contained" startIcon={<CloudUploadIcon /> }>
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
