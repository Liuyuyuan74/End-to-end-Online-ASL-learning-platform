import * as React from 'react';
import CssBaseline from '@mui/material/CssBaseline';
import Grid from '@mui/material/Grid';
import Container from '@mui/material/Container';
import { createTheme, ThemeProvider } from '@mui/material/styles';
import MainFeaturedPost from './MainFeaturedPost';
import FeaturedPost from './FeaturedPost';
import Footer from './Footer';
import helloImg from './img/hello.jpg'

const mainFeaturedPost = {
  title: 'Welcome to Your ASL Translator Website',
  description:
    "Uploading the images of your gesture, getting the meanings of them",
  image: helloImg,
  imageText: 'main image description',
};

const featuredPosts = [
  {
    title: 'Upload Your Images of Gesture:',
  },
];

// TODO remove, this demo shouldn't need to reset the theme.
const defaultTheme = createTheme();

export default function Blog() {
  return (
    <ThemeProvider theme={defaultTheme}>
      <CssBaseline />
      <Container maxWidth="lg">
        <main>
          <MainFeaturedPost post={mainFeaturedPost} />
          <Grid container spacing={4} justifyContent="center" >
            {featuredPosts.map((post) => (
              <FeaturedPost key={post.title} post={post} />
            ))}
          </Grid>
        </main>
      </Container>
      <Footer
        title="WEB-BASED AMERICAN SIGN LANGUAGE TRANSLATOR"
        description="Team 6"
      />
    </ThemeProvider>
  );
}
