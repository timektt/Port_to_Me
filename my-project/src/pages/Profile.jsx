import { useState, useRef } from "react";
import { getAuth, updateProfile } from "firebase/auth";
import { getDownloadURL, ref, uploadBytes } from "firebase/storage";
import { storage } from "../firebase/firebase-config";

export default function Profile() {
  const auth = getAuth();
  const user = auth.currentUser;

  const [name, setName] = useState(user?.displayName || "");
  const [photoURL, setPhotoURL] = useState(user?.photoURL || "/default.jpg");
  const [uploading, setUploading] = useState(false);
  const fileInputRef = useRef();

  const handleUpload = async (file) => {
    if (!file || !user) return;
    setUploading(true);
    const storageRef = ref(storage, `profile-images/${user.uid}`);
    await uploadBytes(storageRef, file);
    const downloadURL = await getDownloadURL(storageRef);
    setPhotoURL(downloadURL);
    setUploading(false);
  };

  const handleSave = async () => {
    await updateProfile(user, {
      displayName: name,
      photoURL: photoURL,
    });
    alert("✅ Profile updated!");
  };

  return (
    <div className="max-w-md mx-auto mt-10 p-14">
      <h2 className="text-2xl font-bold mb-6 text-center">  Profile</h2>

      <div className="mb-4 text-center">
        <img
          src={photoURL}
          alt="Profile"
          className="w-24 h-24 rounded-full mx-auto mb-2 object-cover border"
        />
        <button
          onClick={() => fileInputRef.current.click()}
          className="text-sm hover:underline"
        >
          {uploading ? "Upload" : "Change Photo"}
        </button>
        <input
          type="file"
          accept="image/*"
          className="hidden"
          ref={fileInputRef}
          onChange={(e) => handleUpload(e.target.files[0])}
        />
      </div>

      <input
        type="text"
        placeholder="Your name"
        value={name}
        onChange={(e) => setName(e.target.value)}
        className="w-full mb-3 p-2 rounded bg-gray-800"
      />
      <p className="text-sm mb-4">Email: {user?.email}</p>

      <button
        onClick={handleSave}
        className="w-full bg-green-600 hover:bg-green-700 text-white py-2 rounded"
      >
        Save Changes
      </button>
    </div>
  );
}
