#include <math.h>
#include <stdlib.h>

#if defined(__APPLE__)
  #include <OpenGL/gl.h>
  #include <OpenGL/glu.h>
  #include <GLUT/glut.h>
#else
  #if defined(WIN32) || defined(_WIN32) || defined(__WIN32__)
    #include <windows.h>
  #endif
  #include <GL/gl.h>
  #include <GL/glu.h>
  #include <GL/glut.h>
#endif

#include <map>
#include <set>
#include <vector>
#include <cassert>
#include <memory>
#include <iostream>
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

#ifndef M_PI
  #define M_PI 3.14159265359
#endif

struct Vector {
  float x, y, z;

  Vector(float v = 0) : x(v), y(v), z(v) { }
  Vector(float x, float y, float z) : x(x), y(y), z(z) { }
  Vector operator+(const Vector& v) const { return Vector(x + v.x, y + v.y, z + v.z); }
  Vector operator-(const Vector& v) const { return Vector(x - v.x, y - v.y, z - v.z); }
  Vector operator*(const Vector& v) const { return Vector(x * v.x, y * v.y, z * v.z); }
  Vector operator/(const Vector& v) const { return Vector(x / v.x, y / v.y, z / v.z); }
  friend Vector operator+(float f, const Vector& v) { return v+f; }
  friend Vector operator-(float f, const Vector& v) { return Vector(f)-v; }
  friend Vector operator*(float f, const Vector& v) { return v*f; }
  friend Vector operator/(float f, const Vector& v) { return Vector(f)/v; }
  Vector& operator+=(const Vector& v) { x += v.x, y += v.y, z += v.z; return *this; }
  Vector& operator-=(const Vector& v) { x -= v.x, y -= v.y, z -= v.z; return *this; }
  Vector& operator*=(const Vector& v) { x *= v.x, y *= v.y, z *= v.z; return *this; }
  Vector& operator/=(const Vector& v) { x /= v.x, y /= v.y, z /= v.z; return *this; }
  Vector operator-() const { return Vector(-x, -y, -z); }
  float dot(const Vector& v) const { return x*v.x + y*v.y + z*v.z; }
  friend float dot(const Vector& a, const Vector& b) { return a.dot(b); }
  Vector cross(const Vector& v) const { return Vector(y*v.z - z*v.y, z*v.x - x*v.z, x*v.y - y*v.x); }
  friend Vector cross(const Vector& a, const Vector& b) { return a.cross(b); }
  float length() const { return sqrt(x*x + y*y + z*z); }
  Vector normalize() const { float l = length(); if(l > 1e-3) { return (*this/l); } else { return Vector(); } }
  bool isNull() const { return length() < 1e-3; }
};

enum ControllKeys {W, A, S, D, keys_num};
bool keys_down[keys_num];

struct Camera {
  Vector fwd, pos;
  const float speed, mouse_speed;

  Camera(float speed = 5, float mouse_speed = 0.002f) : fwd(Vector(9, 0, 4).normalize()), pos(-9, 5, -4), speed(speed), mouse_speed(mouse_speed) { }

  void updatePos(float dt) {
    Vector up = Vector(0, 1, 0), right = cross(fwd, up).normalize();
    up = cross(right, fwd).normalize();

    if(keys_down[W] && !keys_down[S]) {
      pos += fwd * speed * dt;
    } else if(keys_down[S] && !keys_down[W]) {
      pos -= fwd * speed * dt;
    }

    if(keys_down[D] && !keys_down[A]) {
      pos += right * speed * dt;
    } else if(keys_down[A] && !keys_down[D]) {
      pos -= right * speed * dt;
    }
  }

  void updateDir(int dx, int dy) {
    Vector y_axis = Vector(0, 1, 0), right = cross(fwd, y_axis).normalize();
    Vector up = cross(right, fwd).normalize();

    // Ha teljesen felfele / lefele néznénk, akkor ne forduljon át a kamera
    float dot_up_fwd = dot(y_axis, fwd);
    if(dot_up_fwd > 0.95f && dy > 0) {
      dy = 0;
    }
    if(dot_up_fwd < -0.95f && dy < 0) {
      dy = 0;
    }

    // Módosítsuk az nézeti irányt
    fwd += mouse_speed * (right * dx + up * dy);
    fwd = fwd.normalize();
  }

  void applyMatrix() const {
    gluLookAt(pos.x, pos.y, pos.z, pos.x+fwd.x, pos.y+fwd.y, pos.z+fwd.z, 0, 1, 0);
  }
} camera;

void glVertex3f(const Vector& v) {
  glVertex3f(v.x, v.y, v.z);
}

void glQuad(const Vector& a, const Vector& b, const Vector& c, const Vector& d) {
  Vector normal = cross(b-a, c-a).normalize();
  glNormal3f(normal.x, normal.y, normal.z);

  int res = 10; // resolution
  for(int i = 0; i < res; i++) {
    for(int j = 0; j < res; j++) {
      glVertex3f(a + i*(b-a)/res     + j*(d-a)/res);
      glVertex3f(a + (i+1)*(b-a)/res + j*(d-a)/res);
      glVertex3f(a + (i+1)*(b-a)/res + (j+1)*(d-a)/res);
      glVertex3f(a + i*(b-a)/res     + (j+1)*(d-a)/res);
    }
  }
}

void drawCube(const Vector& size) {
  glBegin(GL_QUADS); {
    /*       (E)-----(A)
             /|      /|
            / |     / |
          (F)-----(B) |
           | (H)---|-(D)
           | /     | /
           |/      |/
          (G)-----(C)        */

    Vector s = size / 2;

    Vector A(+s.x, +s.y, -s.z), B(+s.x, +s.y, +s.z), C(+s.x, -s.y, +s.z), D(+s.x, -s.y, -s.z),
           E(-s.x, +s.y, -s.z), F(-s.x, +s.y, +s.z), G(-s.x, -s.y, +s.z), H(-s.x, -s.y, -s.z);

    glQuad(A, B, C, D); glQuad(E, H, G, F); glQuad(A, E, F, B);
    glQuad(D, C, G, H); glQuad(B, F, G, C); glQuad(A, D, H, E);

  } glEnd();
}

GLfloat black[4] = {0, 0, 0, 1};

void setSun() {
  {
    float p[4] = {-1, 0.8, -0.6, 0};
    glLightfv(GL_LIGHT7, GL_POSITION, p);

    float c[4] = {0.5, 0.5, 0.5, 1};
    glLightfv(GL_LIGHT7, GL_DIFFUSE, c);
    glLightfv(GL_LIGHT7, GL_AMBIENT, black);

    glEnable(GL_LIGHT7);
  }

  {
    float p[4] = {1, 0.8, 0.6, 0};
    glLightfv(GL_LIGHT6, GL_POSITION, p);

    float c[4] = {0.2, 0.2, 0.2, 1};
    glLightfv(GL_LIGHT6, GL_DIFFUSE, c);
    glLightfv(GL_LIGHT6, GL_AMBIENT, black);

    glEnable(GL_LIGHT6);
  }

}


void setLighting(int i) {
  float p[4] = {0.0f, 0.0f, 0.0f, 1};
  glLightfv(GL_LIGHT0 + i, GL_POSITION, p);
  glLightf(GL_LIGHT0 + i, GL_CONSTANT, 0.0f);
  glLightf(GL_LIGHT0 + i, GL_QUADRATIC_ATTENUATION, 1.0f);

  GLfloat c[4] = {0.3f, 0.3f, 0.3f, 1};
  glLightfv(GL_LIGHT0 + i, GL_DIFFUSE, c);
  glLightfv(GL_LIGHT0 + i, GL_AMBIENT, black);
  glEnable(GL_LIGHT0 + i);
}

GLUquadric *quad = gluNewQuadric();

struct Tree {
  float size;
  Vector pos;
  bool valid;

  void draw() {
    glPushMatrix(); {
      glTranslatef(pos.x, pos.y, pos.z);

      glColor3f(0x53/255., 0x35/255., 0x0A/255.);
      glPushMatrix(); {
        glRotatef(-90, 1, 0, 0);
        gluCylinder(quad, size/3, size/3, size, 8, 8);
      } glPopMatrix();

      glColor3f(0.01, 0.8, 0.12);
      glPushMatrix(); {
        glTranslatef(0, size, 0);
        glRotatef(-90, 1, 0, 0);
        glutSolidCone(size/2, 2*size, 8, 8);
      } glPopMatrix();
    } glPopMatrix();
  }
};

struct Forest {
  Tree trees[13][13];
  Forest() {
    for(int i = 0; i < 13; i++)
      for(int j = 0; j < 13; j++) {
        float x = i*15 - 90, y = j*15 - 90;
        if(fabs(x) < 50 && fabs(y) < 50) {
          trees[i][j].valid = false;
          continue;
        }
        trees[i][j].size = 4 + 3.0f*rand()/RAND_MAX;
        trees[i][j].pos = Vector(x + 10.0f*rand()/RAND_MAX - 5, 0, y + 10.0f*rand()/RAND_MAX - 5);
        trees[i][j].valid = true;
      }
  }

  void draw() {
    for(int i = 0; i < 13; i++) {
      for(int j = 0; j < 13; j++) {
        if(trees[i][j].valid)
          trees[i][j].draw();
      }
    }
  }
} forest;

void drawGround() {
  glColor3f(0.01, 0.8, 0.12);

  glPushMatrix(); {
    glTranslatef(0, -1.0f, 0);
    glScalef(100, 0.5, 100);
    drawCube(2.0f);
  } glPopMatrix();

  forest.draw();
}

void drawPantheon() {
  drawGround();

  glColor3f(1, 1, 1);

  glPushMatrix(); {
    glTranslatef(0, -1.0f, 0);
    glScalef(30, 1, 30);
    drawCube(2.0f);

    glTranslatef(0, 20, 0);
    drawCube(2.0f);

  } glPopMatrix();

  glColor3f(0.2f, 0.4f, 0.6f);

  glColor3f(1, 1, 1);

  int num_cylinders = 6;
  for(int i = 0; i < num_cylinders; ++i) {
    for(int j = 0; j < 4; j++) {
      int sign = j%2 ? 1 : -1;

      glPushMatrix(); {
        if(j < 2) {
          glTranslatef(sign * 25, 0, sign * (50.0f*i/num_cylinders - 25));
        } else {
          glTranslatef(-sign * (50.0f*i/num_cylinders - 25), 0, sign * 25);
        }

        glRotatef(-90, 1, 0, 0);
        gluCylinder(quad, 2, 2, 20, 8, 8);
      } glPopMatrix();
    }
  }
}

using Face = std::vector<size_t>;
using FaceIdx = size_t;
using VertexIdx = size_t;
using VertexPair = std::pair<VertexIdx, VertexIdx>;
using FaceLocalVertexIdx = size_t;
using EdgeIdx = size_t;
using Offset = size_t;

struct HalfEdge {
  VertexIdx start_vertex_idx, end_vertex_idx;
  FaceLocalVertexIdx start_vertex_face_idx, end_vertex_face_idx;
  FaceIdx face_idx;
};

using VertexFan = std::vector<HalfEdge>;

struct Edge {
  VertexIdx start_vertex_idx, end_vertex_idx;
  FaceIdx first_face_idx, second_face_idx;
};

VertexPair MakeVertexPair(VertexIdx a, VertexIdx b) {
  return std::make_pair(std::min(a, b), std::max(a, b));
}

Vector CalculateNormal(const Face& face, const std::vector<Vector>& vertices) {
  if (face.size() < 3) {
    return Vector(0, 0, 0);
  } else if (face.size() <= 4) {
    auto vertex0 = vertices[face[0]];
    auto vertex1 = vertices[face[1]];
    auto vertexn = vertices[face[face.size()-1]];
    return cross((vertex1-vertex0).normalize(), (vertexn-vertex0).normalize()).normalize();
  } else {
    Vector center(0, 0, 0);
    for (VertexIdx index : face) {
      center += vertices[index];
    }
    center /= face.size();

    Vector normal(0, 0, 0);
    for (int i = 0; i < face.size(); ++i) {
      Vector a = (vertices[face[i]] - center).normalize();
      Vector b = (vertices[face[(i+1)%face.size()]] - center).normalize();
      normal += cross(a, b);
    }
    return normal.normalize();
  }
}

class Mesh {
public:
  Mesh() = default;
  Mesh(const std::string& filename) {
    Assimp::Importer importer;
    const aiScene* scene{importer.ReadFile(
      filename.c_str(), aiProcess_JoinIdenticalVertices)};

    for (unsigned mesh_idx = 0; mesh_idx < scene->mNumMeshes; ++mesh_idx) {
      const aiMesh* mesh = scene->mMeshes[mesh_idx];
      assert(!mesh->HasNormals());
      Offset vertex_index_start = vertices_.size();
      for (unsigned vertex_idx = 0; vertex_idx < mesh->mNumVertices; ++vertex_idx) {
        auto vertex = mesh->mVertices[vertex_idx];
        vertices_.push_back({vertex.x, vertex.y, vertex.z});
      }

      for (FaceIdx face_idx = 0; face_idx < mesh->mNumFaces; face_idx++) {
        const aiFace& face = mesh->mFaces[face_idx];
        faces_.push_back({});
        for (int index_idx = 0; index_idx < face.mNumIndices; ++index_idx) {
          faces_.back().push_back(vertex_index_start + face.mIndices[index_idx]);
        }
        normals_.push_back(CalculateNormal(faces_.back(), vertices_));
      }
    }
  }

  void Draw() const {
    for (int face_idx = 0; face_idx < faces_.size(); ++face_idx) {
      const Face& face = faces_[face_idx];
      glBegin(GL_TRIANGLE_FAN); {
        auto normal = normals_[face_idx];
        glNormal3f(normal.x, normal.y, normal.z);
        Vector center(0, 0, 0);
        // for (VertexIdx index : face) {
        //   center += vertices_[index];
        // }
        // center /= face.size();
        for (VertexIdx index : face) {
          auto vertex = vertices_[index] + center/64.0;
          glVertex3f(vertex.x, vertex.y, vertex.z);
        }
      } glEnd();
    }
  }

  void UpdateCache() const {
    if (!edges_.empty()) {
      std::cout << "Update cache called with cache up to date" << std::endl;
      return;
    }
    vertex_fans_.resize(vertices_.size());

    std::map<std::pair<VertexIdx, VertexIdx>, FaceIdx> edge_to_face;
    for (FaceIdx face_idx = 0; face_idx < faces_.size(); ++face_idx) {
      Face face = faces_[face_idx];
      int n = face.size();
      for (int i = 0; i < n; ++i) {
        VertexPair edge = MakeVertexPair(face[i], face[(i+1) % n]);
        auto iter = edge_to_face.find(edge);
        if (iter == edge_to_face.end()) {
          edge_to_face.insert(std::make_pair(edge, face_idx));
        } else {
          Edge finalEdge;
          finalEdge.start_vertex_idx = edge.first;
          finalEdge.end_vertex_idx = edge.second;
          finalEdge.first_face_idx = iter->second;
          finalEdge.second_face_idx = face_idx;
          edges_.push_back(finalEdge);
          edge_map_.insert(std::make_pair(edge, edges_.size() - 1));
        }

        vertex_fans_[face[i]].push_back({
          face[i], face[(i+1) % n], size_t(i), size_t((i+1) % n), face_idx});
      }
    }

    // order vertex_fans
    for (int j = 0; j < vertex_fans_.size(); ++j) {
      VertexFan* vertex_fan = &vertex_fans_[j];
      if (vertex_fan->size() < 3) {
        continue;
      }
      VertexFan old_fan = *vertex_fan;
      VertexFan new_fan;
      new_fan.push_back(vertex_fan->back());
      vertex_fan->pop_back();

      while (!vertex_fan->empty()) {
        VertexIdx last_end_idx = faces_[new_fan.back().face_idx][new_fan.back().end_vertex_face_idx];

        bool found = false;
        for (size_t i = 0; i < vertex_fan->size(); ++i) {
          // check if this face contains the last end idx

          const Face& face = faces_[(*vertex_fan)[i].face_idx];
          for (VertexIdx index : face) {
            if (index == last_end_idx) {
              new_fan.push_back((*vertex_fan)[i]);
              vertex_fan->erase(vertex_fan->begin() + i);
              found = true;
              break;
            }
          }
        }

#ifdef ENSUREMANIFOLD
        assert(found);
#endif
        if (!found) {
          break;
        }
      }

#ifdef ENSUREMANIFOLD
      assert(new_fan.size() >= 3);
#endif
      if (new_fan.size() >= 3) {
        *vertex_fan = new_fan;
      } else {
        *vertex_fan = {}; //old_fan;
      }
    }
  }

  Mesh DooSabin(int repeat = 1) const {
    assert(repeat >= 1);
    UpdateCache();

    Mesh mesh;
    for (int face_idx = 0; face_idx < faces_.size(); ++face_idx) {
      Face face = faces_[face_idx];
      Face vertex_face;

      if (face.size() < 3) {
        continue;
      }

      Offset face_start_offset = mesh.vertices_.size();
      face_to_start_offset_.insert(std::make_pair(face_idx, face_start_offset));

      for (int i = 0; i < face.size(); ++i) {
        {
          Vector vertex (0, 0, 0);
          double n = face.size();

          for (int j = 0; j < face.size(); ++j) {
            if (i == j) {
              double alpha = (n + 5) / (4*n);
              vertex += alpha * vertices_[face[j]];
            } else {
              double cosarg = (2 * M_PI * (i-j)) / n;
              double alpha = (3 + 2*cos(cosarg)) / (4*n);
              vertex += alpha * vertices_[face[j]];
            }
          }

          mesh.vertices_.push_back(vertex);
          vertex_face.push_back(mesh.vertices_.size() - 1);
        }

        size_t n = face.size();
        auto iter = edge_map_.find(MakeVertexPair(face[i], face[(i+1) % n]));
        if (iter != edge_map_.end()) {
          Edge edge = edges_[iter->second];
          if (face_idx == edge.first_face_idx) {
            continue; // we only want to proccess the second one
          }

          Face edge_face;
          Face first_face = faces_[edge.first_face_idx];
          assert(face_to_start_offset_.find(edge.first_face_idx) != face_to_start_offset_.end());

          Offset first_face_start_offset = face_to_start_offset_[edge.first_face_idx];
          for (int k = 0; k < first_face.size(); ++k) {
            if (first_face[k] == face[i]) {
              edge_face.push_back(first_face_start_offset + k);
              int index_on_face = k == 0 ? first_face.size()-1 : k-1;
              assert(first_face[index_on_face] == face[(i+1) % n]);
              if (first_face[index_on_face] != face[(i+1) % n]) {
                index_on_face = (k + 1) % first_face.size();
              }
              assert(first_face[index_on_face] == face[(i+1) % n]);
              edge_face.push_back(first_face_start_offset + index_on_face);
              break;
            }
          }

          edge_face.push_back(face_start_offset + ((i+1) % n));
          edge_face.push_back(face_start_offset + i);

          mesh.faces_.push_back(edge_face);
          mesh.normals_.push_back(CalculateNormal(edge_face, mesh.vertices_));
        } else {
#ifdef ENSUREMANIFOLD
          assert(false);
#endif
        }

      }
      mesh.faces_.push_back(vertex_face);
      mesh.normals_.push_back(CalculateNormal(face, vertices_));
    }

    for (const VertexFan& vertex_fan : vertex_fans_) {
#ifdef ENSUREMANIFOLD
      assert (!vertex_fan.empty());
#endif
      if (vertex_fan.empty()) {
        continue;
      }

      Face vertex_face;
      for (int i = vertex_fan.size() - 1; i >= 0; --i) {
        HalfEdge half_edge = vertex_fan[i];
        auto offset_iter = face_to_start_offset_.find(half_edge.face_idx);
#ifdef ENSUREMANIFOLD
        assert(offset_iter != face_to_start_offset_.end());
#endif
        if (offset_iter == face_to_start_offset_.end()) {
          continue;
        }
        Offset offset = offset_iter->second;
        vertex_face.push_back(offset + half_edge.start_vertex_face_idx);
      }
      if (vertex_face.size() < 3) {
        continue;
      }

      mesh.faces_.push_back(vertex_face);
      Vector normal = CalculateNormal(vertex_face, mesh.vertices_);
      mesh.normals_.push_back(normal);
    }

    if (repeat == 1) {
      return mesh;
    } else {
      return mesh.DooSabin(repeat - 1);
    }
  }

  Mesh CatmulClark(int repeat = 1) const {
    assert(repeat >= 1);
    Mesh mesh;
    mesh.vertices_ = vertices_;

    UpdateCache();

    for (FaceIdx face_idx = 0; face_idx < faces_.size(); face_idx++) {
      const Face& face = faces_[face_idx];
      Vector center(0, 0, 0);
      for (VertexIdx index : face) {
        center += mesh.vertices_[index];
      }
      center /= face.size();
      mesh.vertices_.push_back(center);
      mesh.face_centers_[face_idx] = mesh.vertices_.size() - 1;
    }

    for (EdgeIdx edge_idx = 0; edge_idx < edges_.size(); edge_idx++) {
      const Edge& edge = edges_[edge_idx];
      const Vector& a = mesh.vertices_[edge.start_vertex_idx];
      const Vector& b = mesh.vertices_[edge.end_vertex_idx];
      assert(mesh.face_centers_.find(edge.first_face_idx) != mesh.face_centers_.end() &&
             mesh.face_centers_.find(edge.second_face_idx) != mesh.face_centers_.end());
      const Vector& c = mesh.vertices_[mesh.face_centers_[edge.first_face_idx]];
      const Vector& d = mesh.vertices_[mesh.face_centers_[edge.second_face_idx]];
      mesh.vertices_.push_back((a + b + c + d) / 4.0);
      mesh.edge_centers_[edge_idx] = mesh.vertices_.size() - 1;
    }

    for (const VertexFan& vertex_fan : vertex_fans_) {
#ifdef ENSUREMANIFOLD
      assert(vertex_fan.size() >= 3);
#endif
      if (vertex_fan.size() < 3) {
        continue;
      }

      Vector& vertex = mesh.vertices_[vertex_fan[0].start_vertex_idx];

      Vector face_vertex_contrib;
      Vector edge_vertex_contrib;
      double n = 0;
      for (const HalfEdge& half_edge : vertex_fan) {
        const Face& face = faces_[half_edge.face_idx];
        auto face_center_iter = mesh.face_centers_.find(half_edge.face_idx);
#ifdef ENSUREMANIFOLD
        assert (face_center_iter != mesh.face_centers_.end());
#endif
        if (face_center_iter == mesh.face_centers_.end()) {
          continue;
        }
        VertexIdx face_center_idx = face_center_iter->second;
        VertexPair pair = MakeVertexPair(half_edge.start_vertex_idx,
                                         half_edge.end_vertex_idx);
        auto edge_iter = edge_map_.find(pair);
#ifdef ENSUREMANIFOLD
        assert (edge_iter != edge_map_.end());
#endif
        if (edge_iter == edge_map_.end()) {
          continue;
        }

        EdgeIdx edge_idx = edge_iter->second;
        VertexIdx edge_center_idx = mesh.edge_centers_[edge_idx];

        face_vertex_contrib += mesh.vertices_[face_center_idx];
        edge_vertex_contrib += mesh.vertices_[edge_center_idx];
        n++;
      }

      if (n < 3) {
        continue;
      }
      vertex = 1.0/(n*n) * face_vertex_contrib +
               2.0/(n*n) * edge_vertex_contrib +
               (n - 3) / n * vertex;
    }

    for (FaceIdx face_idx = 0; face_idx < faces_.size(); face_idx++) {
      const Face& face = faces_[face_idx];
      size_t n = face.size();
      auto face_center_iter = mesh.face_centers_.find(face_idx);
#ifdef ENSUREMANIFOLD
      assert (face_center_iter != mesh.face_centers_.end());
#endif
      if (face_center_iter == mesh.face_centers_.end()) {
        continue;
      }
      VertexIdx face_center = face_center_iter->second;
      for (VertexIdx i = 0; i < n; i++) {
        VertexIdx end_vertex = face[(i+1)%n];
        VertexPair first_pair = MakeVertexPair(face[i], face[(i+1)%n]);
        VertexPair second_pair = MakeVertexPair(face[(i+1)%n], face[(i+2)%n]);
        auto first_edge_iter = edge_map_.find(first_pair);
        auto second_edge_iter = edge_map_.find(second_pair);
#ifdef ENSUREMANIFOLD
        assert (first_edge_iter != edge_map_.end() &&
                second_edge_iter != edge_map_.end());
#endif
        if (first_edge_iter == edge_map_.end() ||
            second_edge_iter == edge_map_.end()) {
          continue;
        }
        EdgeIdx first_edge = first_edge_iter->second;
        EdgeIdx second_edge = second_edge_iter->second;
        assert(mesh.edge_centers_.find(first_edge) != mesh.edge_centers_.end() &&
               mesh.edge_centers_.find(second_edge) != mesh.edge_centers_.end());
        VertexIdx first_edge_vertex = mesh.edge_centers_[first_edge];
        VertexIdx second_edge_vertex = mesh.edge_centers_[second_edge];
        const auto& face = {face_center, first_edge_vertex, end_vertex, second_edge_vertex};
        mesh.faces_.emplace_back(face);
        mesh.normals_.push_back(CalculateNormal(mesh.faces_.back(), mesh.vertices_));
      }
    }

    if (repeat == 1) {
      return mesh;
    } else {
      return mesh.CatmulClark(repeat - 1);
    }
  }

  Mesh HalfDivision(int repeat = 1) const {
    assert(repeat >= 1);
    UpdateCache();

    Mesh mesh;

    for (const Face& face : faces_) {
      Face new_face;
      for (VertexIdx vertex_idx = 0; vertex_idx < face.size(); vertex_idx++) {
        VertexPair pair = MakeVertexPair(face[vertex_idx],
                                         face[(vertex_idx + 1) % face.size()]);
        const Vector& a = vertices_[pair.first];
        const Vector& b = vertices_[pair.second];
        mesh.vertices_.push_back((a + b) / 2.0);
        new_face.push_back(mesh.vertices_.size() - 1);
        auto edge_iter = edge_map_.find(pair);
#ifdef ENSUREMANIFOLD
        assert(edge_iter != edge_map_.end())
#endif
        if (edge_iter != edge_map_.end()) {
          mesh.edge_centers_[edge_iter->second] = mesh.vertices_.size() - 1;
        }
      }

      mesh.faces_.push_back(new_face);
      mesh.normals_.push_back(CalculateNormal(new_face, mesh.vertices_));
    }

    for (const VertexFan& vertex_fan : vertex_fans_) {
#ifdef ENSUREMANIFOLD
      assert (!vertex_fan.empty());
#endif
      if (vertex_fan.empty()) {
        continue;
      }

      Face vertex_face;
      for (int i = vertex_fan.size() - 1; i >= 0; --i) {
        const HalfEdge& half_edge = vertex_fan[i];
        VertexPair pair = MakeVertexPair(half_edge.start_vertex_idx,
                                         half_edge.end_vertex_idx);
        auto edge_iter = edge_map_.find(pair);
        assert(edge_iter != edge_map_.end());
        if (edge_iter == edge_map_.end()) {
          continue;
        }
        EdgeIdx edge_idx = edge_iter->second;
        VertexIdx edge_center = edge_centers_[edge_idx]; // todo assert
        vertex_face.push_back(edge_center);
      }
      if (vertex_face.size() < 3) {
        continue;
      }

      mesh.faces_.push_back(vertex_face);
      mesh.normals_.push_back(CalculateNormal(vertex_face, mesh.vertices_));
    }


    if (repeat == 1) {
      return mesh;
    } else {
      return mesh.HalfDivision(repeat - 1);
    }
  }

private:
  std::vector<Vector> vertices_;
  std::vector<Face> faces_;
  std::vector<Vector> normals_;

  mutable std::vector<Edge> edges_;
  mutable std::vector<VertexFan> vertex_fans_;
  mutable std::map<FaceIdx, VertexIdx> face_centers_;
  mutable std::map<EdgeIdx, VertexIdx> edge_centers_;
  mutable std::map<VertexPair, EdgeIdx> edge_map_;
  mutable std::map<FaceIdx, Offset> face_to_start_offset_;
};

// Mesh mesh("stuff.obj");
// Mesh mesh("cube.obj");
Mesh mesh("cube2.obj");
// Mesh mesh("teapot.obj");
// Mesh mesh("teddy.obj");
// Mesh mesh("humanoid_quad.obj");
// Mesh mesh("skyscraper.obj");
// Mesh mesh("al.obj");
// Mesh mesh("cessna.obj");

// Mesh mesh2(mesh.DooSabin(2));
// Mesh mesh2(mesh.CatmulClark(2));
Mesh mesh2(mesh.HalfDivision());

void onDisplay() {
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();

  camera.applyMatrix();
  setSun();

  glPushMatrix(); {
    // glEnable(GL_CULL_FACE);
    glTranslatef(0, 5.0f, 0);
    // glScalef(0.2f, 0.2f, 0.2f);
    glColor3f(1, 0, 0);
    mesh.Draw();

    glTranslatef(8.0f, 0.0f, 0);
    glColor3f(0.2, 0.4, 0.6);
    mesh2.Draw();
    glDisable(GL_CULL_FACE);
  } glPopMatrix();

  drawPantheon();

  glutSwapBuffers();
}

void onIdle() {
  static float last_time = glutGet(GLUT_ELAPSED_TIME);
  float time = glutGet(GLUT_ELAPSED_TIME);
  float dt = (time - last_time) / 1000.0f;
  last_time = time;

  camera.updatePos(dt);
  glutPostRedisplay();
}

void onInitialization() {
  glClearColor(135./255., 206./255., 235./255., 1);
  glEnable(GL_DEPTH_TEST);
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  gluPerspective(60, 1, 0.2, 200);
  glMatrixMode(GL_MODELVIEW);

  glFrontFace(GL_CCW);
  glCullFace(GL_BACK);

  glEnable(GL_LIGHTING);
  glEnable(GL_COLOR_MATERIAL);
  glEnable(GL_SMOOTH);
}

void onReshape(int w, int h) {
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  gluPerspective(60, float(h)/float(w), 0.2, 200);
  glMatrixMode(GL_MODELVIEW);
}

void onKeyboard(unsigned char key, int, int) {
  switch(key) {
    case 'w': case 'W':
      keys_down[W] = true;
      break;
    case 's': case 'S':
      keys_down[S] = true;
      break;
    case 'a': case 'A':
      keys_down[A] = true;
      break;
    case 'd': case 'D':
      keys_down[D] = true;
      break;
  }
}

void onKeyboardUp(unsigned char key, int, int) {
    switch(key) {
    case 'w': case 'W':
      keys_down[W] = false;
      break;
    case 's': case 'S':
      keys_down[S] = false;
      break;
    case 'a': case 'A':
      keys_down[A] = false;
      break;
    case 'd': case 'D':
      keys_down[D] = false;
      break;
  }
}

int last_x, last_y;
void onMouse(int, int, int x, int y) {
  last_x = x;
  last_y = y;
}

void onMouseMotion(int x, int y) {
  camera.updateDir(x-last_x, last_y-y);
  last_x = x;
  last_y = y;
}

int main(int argc, char **argv) {
  glutInit(&argc, argv);
  glutInitWindowSize(600, 600);
  glutInitWindowPosition(100, 100);
  glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);

  glutCreateWindow("Grafika pelda program");

  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();

  onInitialization();

  glutDisplayFunc(onDisplay);
  glutMouseFunc(onMouse);
  glutIdleFunc(onIdle);
  glutKeyboardFunc(onKeyboard);
  glutKeyboardUpFunc(onKeyboardUp);
  glutMotionFunc(onMouseMotion);
  // glutReshapeFunc(onReshape);

  glutMainLoop();

  return 0;
}
