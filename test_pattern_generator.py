# Lauri Varjo, 2024

import numpy as np

def rot_mat(angle):
        # Creates a rotation matrix with angle
        angle = np.radians(angle)
        return np.array(((np.cos(angle), -np.sin(angle)),(np.sin(angle),  np.cos(angle))))

def rotate(src_img, angle, method="nn", k=1, bg_color=0, rot_org="center"):
        # Rotates an input image by the given angle

        src_dim = src_img.shape

        ofs_y = 0.5 if src_dim[0] % 2 == 0 else 1.5
        ofs_x = 0 if src_dim[1] % 2 == 0 else 1
        if rot_org == "center":
            x = np.arange(-src_dim[1]//2+ofs_x,src_dim[1]//2+ofs_x,1)
            y = np.arange(-src_dim[0]//2+ofs_y,src_dim[0]//2+ofs_y,1)
        elif rot_org == "corner":
            x = np.arange(0,src_dim[1],1)
            y = np.arange(0,src_dim[0],1)

        X,Y = np.meshgrid(x,y)

        dest_xy = np.array([X.flatten(),Y.flatten()])

        R = rot_mat(angle)

        src_xy = np.transpose(dest_xy) @ R

        dest_img = np.zeros(src_img.shape)
        count=0
        for src_x,src_y in src_xy:
            row = count // src_dim[1]
            col = count - src_dim[0]*row

            # Scale indices back to 0->src_dim
            y_ind = src_y+src_dim[1]//2
            x_ind = src_x+src_dim[0]//2

            if method == "knn":
                # K-nearest neighbor, TODO
                # patch = np.zeros((k,k))
                # src_patch = np.zeros([k+2,k+2])

                y_lower = 0 if y_ind-k < 0 else y_ind-k
                y_upper = src_dim[1] if y_ind+k > src_dim[1] else y_ind+k
                x_lower = 0 if x_ind-k < 0 else x_ind-k
                x_upper = src_dim[0] if x_ind+k > src_dim[0] else x_ind+k

                # print(y_lower, y_upper, x_lower, x_upper)
                src_patch = src_img[int(y_lower):int(y_upper),int(x_lower):int(x_upper)]
                dest_img[row,col] = np.mean(src_patch)

            elif method == "linear":
                y1 = np.floor(y_ind).astype(int)
                y2 = np.ceil(y_ind).astype(int)
                x1 = np.floor(x_ind).astype(int)
                x2 = np.ceil(x_ind).astype(int)

                v11 = src_img[y1, x1] if 0 < y1 < src_dim[0] and 0 < x1 < src_dim[1] else 0
                v12 = src_img[y1, x2] if 0 < y1 < src_dim[0] and 0 < x2 < src_dim[1] else 0
                v21 = src_img[y2, x1] if 0 < y2 < src_dim[0] and 0 < x1 < src_dim[1] else 0
                v22 = src_img[y2, x2] if 0 < y2 < src_dim[0] and 0 < x2 < src_dim[1] else 0

                # With matrix multiplication
                # rat = 1/ np.max([((x2-x1)*(y2-y1)),1])
                # x_mat = np.array([x2 - x_ind, x_ind-x1])
                # val_mat = np.array([[v11, v12], [v21, v22]])
                # y_mat = np.array([y2-y_ind, y_ind-y1])

                # dest_img[row,col] = rat*(x_mat@val_mat@y_mat)
                if x1 == x_ind:
                    r1 = v11
                    r2 = v21

                elif x2 == x_ind:
                    r1 = v12
                    r2 = v22

                else:
                    # Basic
                    dx = np.abs(x2-x1)
                    x_ui = np.abs(x2-x_ind) # upper to ind
                    x_li = np.abs(x_ind-x1) # lower to ind

                    r1 = (x_ui/dx)*v11 + (x_li/dx)*v12
                    r2 = (x_ui/dx)*v21 + (x_li/dx)*v22
                
                if y1 == y_ind:
                    val = r1
                else:
                    dy = np.abs(y2-y1)
                    y_ui = np.abs(y2-y_ind) # upper to ind
                    y_li = np.abs(y_ind-y1) # lower to ind
                    val = (y_ui/dy)*r1 + (y_li/dy)*r2

                dest_img[row,col] = val

            else:
                # Direct nearest neighbor
                if src_x > -src_dim[1]//2 and src_y > -src_dim[0]//2 and src_x < src_dim[1]//2 and src_y < src_dim[0]//2:
                # if 0 < x_ind < src_dim[1] and 0 <y_ind < src_dim[0]:
                    # print(src_x, src_dim[1]//2)
                    # print(src_y, -src_dim[0]//2)
                    #print(x_ind, y_ind)
                    try:
                        dest_img[row,col] = src_img[int(y_ind),int(x_ind)]
                    except IndexError:
                        pass #print("out-of-bounds")
                else:
                    try:
                        dest_img[row,col] = bg_color
                    except IndexError:
                        pass #print("out-of-bounds")
            count+=1
        
        return dest_img

def crop(arr, dest_dim, debug=False):
        src_dim = arr.shape
        src_dim = np.array(src_dim)
        dest_dim=np.array(dest_dim)
        
        h1 = src_dim[0]//2-dest_dim[0]//2
        h2 = src_dim[0]//2+dest_dim[0]//2 + (0 if dest_dim[0] % 2 == 0 else 1)
        w1 = src_dim[1]//2-dest_dim[1]//2
        w2 = src_dim[1]//2+dest_dim[1]//2 + (0 if dest_dim[1] % 2 == 0 else 1)

        if debug:
            print("=Inside crop=")
            print("src_dims", src_dim)
            print("dest_dim", dest_dim)
            print("h_limits:", [h1,h2])
            print("w_limits:", [w1,w2])

        return arr[h1:h2,w1:w2]


class TestPatternGenerator():
    """
        Class for generating test patterns for measuring optical systems etc.

        Parameters:
            h,w = target pattern dimensions height and width
            ret_range = the range in which the test images are returned
            ret_type = the datatype for generated test patterns
            channel = return a 2d ("mono") or 3d array ("rgb" or any combination of "r","g","b", "rg","bg" etc.)

    """
    def __init__(self, h, w, ret_range=[0,255], ret_type=np.uint8, channel="mono"):
        self.h = h
        self.w = w
        self.c = int(np.sqrt(self.h**2+self.w**2)+3) #hypotenuse
        self.ret_type = ret_type
        self.ret_range = ret_range
        self.channel = channel

    def out(self, arr):
        # Format output images as specified by the class
        if self.channel == "mono":
            return (arr*self.ret_range[1]).astype(self.ret_type)
        
        if self.channel == "rgb":
            ret_arr = (arr*self.ret_range[1]).astype(self.ret_type)
            ret_arr = np.stack([ret_arr,ret_arr,ret_arr])
            return np.transpose(ret_arr, [1,2,0])
        
        else:
            # print(self.channel)
            img = np.zeros([arr.shape[0], arr.shape[1], 3])
            for ch in self.channel:
                ch_ix = "rgb".find(ch)
                # print(ch_ix)
                if  ch_ix > -1:
                    img[:,:,ch_ix] = arr

            return (img*self.ret_range[1]).astype(self.ret_type)

            
    def unicolor_img(self, val, channel):
        """
        Generates an unicolor image 

        Parameters:
            val = value (0-255 if uint8)
            channel = channel number [0,1,2] as [r,g,b]
        """
        # Generates an unicolor image with given val (typically 0-255) and channel number (tpically 0,1,2 as r,g,b)
        arr = np.zeros([self.h,self.w,3])
        arr[:,:,channel] = val
        return arr.astype(self.ret_type)
    
    def rgb_white(self,val):
        arr = np.zeros([self.h,self.w,3])
        arr[...] = val
        return arr.astype(self.ret_type)
    
    def line_grid(self, n=None, n_div=10, px=1, centered=True, direction="both", orientation=0, 
                  interp=False, drawcolor="white"):
        """
        Generate a line grid 

        Parameters:
            n = draw a line every n pixels
            n_div = instead of n, specify number of lines to draw 
            px = line thickness in pixels
            direction = "horz", "vert" or "both"
            centered = lines cross at the center if set True
            orientation = lines rotation in degrees
            interp = interpolation method if rotated, False (defaults nearest) or "linear"
            drawcolor = specifies the color of drawn lines in "black" or "white", the background will correspond to the other
        """
        
        if n is None:
            n = self.c//n_div

        # print("c:", self.c)
        # print("n:", n)

        arr = np.zeros((self.c,self.c))
        # arr = np.zeros((self.h,self.w))

        if centered:
            n_ofs = n # + 4 #+  (1 if n % 2 != 0 else 0) # +(1 if n % 2 == 0 else 0)
        else:
            n_ofs = n//2

        for px_i in range(px):
            px_ofs = px_i//2
            

            if px_i%2==0:
                if direction in ["vert","both"]:
                    arr[:,px_ofs+n_ofs::n] = 1
                    # arr[:,::n] = 1
                if direction in ["horz","both"]:
                    arr[px_ofs+n_ofs::n,:] = 1
                    # arr[::n,:] = 1
            else:
                if direction in ["vert","both"]:
                    arr[:,n_ofs-px_ofs::n] = 1
                if direction in ["horz","both"]:
                    arr[n_ofs-px_ofs::n,:] = 1

        if drawcolor=="black":
            arr=1-arr

        if orientation != 0:
            arr = rotate(arr,orientation, method=interp)

        arr = crop(arr, [self.h, self.w])

        return self.out(arr)
        # return (arr*self.ret_range[1]).astype(self.ret_type)

    def line_grid2(self, n=None, n_div=10, px=1, centered=True, direction="both", orientation=0, 
                  interp=False, drawcolor="white"):
        """
        Generate a line grid 

        Parameters:
            n = draw a line every n pixels
            n_div = instead of n, specify number of lines to draw 
            px = line thickness in pixels
            direction = "horz", "vert" or "both"
            centered = lines cross at the center if set True
            orientation = lines rotation in degrees
            interp = interpolation method if rotated, False (defaults nearest) or "linear"
            drawcolor = specifies the color of drawn lines in "black" or "white", the background will correspond to the other
        """
        
        if n is None:
            n = self.c//n_div

        # print("c:", self.c)
        # print("n:", n)

        arr = np.zeros((self.c,self.c))
        # arr = np.zeros((self.h,self.w))

        for offset_y in range(0,arr.shape[0]//2,n):
            for offset_x in range(0,arr.shape[1]//2,n):
                # print(offset_y, offset_x, print(arr.shape))
                arr += self.crosshair(px=px, offset=(offset_y,offset_x), unprocessed=True)
                arr[arr>1] = 1

        for offset_y in range(-n,-arr.shape[0]//2,-n):
            for offset_x in range(-n,0-arr.shape[1]//2,-n):
                arr += self.crosshair(px=px, offset=(offset_y,offset_x), unprocessed=True)
                arr[arr>1] = 1

        if drawcolor=="black":
            arr=1-arr

        if orientation != 0:
            arr = rotate(arr,orientation, method=interp)

        arr = crop(arr, [self.h, self.w])

        return self.out(arr)
        # return (arr*self.ret_range[1]).astype(self.ret_type)
    
    def crosshair(self, px=1, orientation=0, interp=False, offset=(0,0), unprocessed=False):
        """
        Generates a crosshair with a given orientation

        Parameters:
            px = thickness in pixels
            orientation = angle in degrees
            interp = interpolation method if rotated, False (defaults nearest) or "linear"
        """
 
        # Create a crosshair with px as width in pixels
        # NOTE: for dimension % 2 == 0, the center is offset to rb

        c = self.c
        arr = np.zeros([c,c])
        arr[c//2+offset[0], :] = 1
        arr[:, c//2+offset[1]] = 1

        for px_i in range(px):
            # print(px_i)
            # print(px_i//2)
            if px_i%2==0:
                arr[c//2+px_i//2+offset[0], :] = 1
                arr[:, c//2+px_i//2+offset[1]] = 1
            else:
                arr[c//2-px_i//2+offset[0], :] = 1
                arr[:, c//2-px_i//2+offset[1]] = 1

        if orientation != 0:
            arr = rotate(arr,orientation, method=interp)

        if unprocessed:
            return arr
        
        # print(arr.shape)
        arr = crop(arr, [self.h, self.w])
        # print(arr.shape)

        
        return self.out(arr)

    def dot_grid(self, rad=1, sep=10, centered=True):
        """
        Generates a dot grid

        Parameters:
            rad = aperture radius of one dot in px
            sep = separation between dots
            centered = center the grid so that one dot lies in the very center
        """

        start_offset = sep//2 if centered else 0
        
        arr = np.zeros((self.h,self.w))
        for off_x in np.arange(-self.w//2,self.w//2+0.01,sep)+start_offset:
            for off_y in np.arange(-self.h//2,self.h//2+0.01,sep)+start_offset:
                arr += self.circular_aperture(rad, offset=(off_x,off_y), unprocessed=True)
                arr[arr>1] = 1

        return self.out(arr)
        

    def circular_aperture(self, rad=1, offset=(0,0), unprocessed=False):
        """
        Generates one circular aperture

        Parameters:
            rad = aperture radius in px
            offset = (x,y) offset in x and y in px
        """

        x = np.arange(-self.w//2,self.w//2,1)+1-offset[0]
        y = np.arange(-self.h//2,self.h//2,1)+1-offset[1]
        X,Y = np.meshgrid(x,y)
        dist = np.sqrt(X**2+Y**2)
        arr = dist<rad
        arr[arr>0] = 1
        arr[arr<0] = 0

        return arr if unprocessed else self.out(arr)
        
    def sineplate(self, rad=None, freq=1, offset=(0,0), binary=False):
        """
        Generates a sine plate with constant frequency

        Parameters:
            rad = crop the plate to the selected radius, use None for no crop
            offset = plate center offset from the image center
            binary = outputs a binary image if True
        """
        x = np.arange(-self.w//2,self.w//2,1)+1-offset[0]
        y = np.arange(-self.h//2,self.h//2,1)+1-offset[1]
        X,Y = np.meshgrid(x,y)
        dist = np.sqrt(X**2+Y**2)

        arr = np.sin(dist/np.pi*freq)

        arr = (arr+1)/2

        if rad is not None:
            arr[dist>rad] = 0

        if binary:
            arr[arr>=0.5] = 1
            arr[arr<0.5] = 0

        return self.out(arr)
    
    def zoneplate(self, f=0.7, rad=None, offset=(0,0), binary=False):
        """
        Generates zoneplate with frequency increasing from center to the edges
            
        Parameters:
            f = defines the frequencies
            rad = crop the plate to the selected radius, use None for no crop
            offset = plate center offset from the image center
            binary = outputs a binary image if True

            Reference: https://se.mathworks.com/matlabcentral/fileexchange/35961-zone-plate-test-image
        """
        x = np.arange(-self.w//2,self.w//2,1)+1-offset[0]
        y = np.arange(-self.h//2,self.h//2,1)+1-offset[1]
        X,Y = np.meshgrid(x,y)
        dist = np.sqrt(X**2+Y**2)
        dist = dist.astype(float)

        km = f*np.pi
        rm = self.w
        w = rm/10
        t1 = np.sin((km*dist**2)/(2*rm))
        t2 = 0.5*np.tanh((rm-dist)/w) + 0.5
        g = t1*t2

        arr = (g+1)/2

        # print(np.min(arr))
        # print(np.max(arr))

        if rad is not None:
            arr[dist>rad] = 0

        if binary:
            arr[arr>=0.5] = 1
            arr[arr<0.5] = 0

        return self.out(arr)
        
    def square_aperture(self, px=4, orientation=2, offset=(0,0), interp=False):
        """
        Generates a square aperture

        Parameters:
            px = diameter in px
            offset = (x,y) offset in x and y in px
        """
        arr = np.zeros((self.h,self.w))
        y1 = self.h//2 + offset[0] - px
        y2 = self.h//2 + offset[0] + px
        x1 = self.w//2 + offset[1] - px
        x2 = self.w//2 + offset[1] + px

        arr[y1:y2,x1:x2] = 1

        if orientation != 0:
            arr = rotate(arr, angle=orientation, method=interp)
            # arr = ndimage.rotate(arr, orientation, reshape=False)#
        
        return self.out(arr)

    def frame(self, pad=3, drawcolor="white"):
        """
        Generates a frame on the edges of width pad

        Parameters:
            pad = thickness in px
        """
        # Frame with pad in px
        arr = np.zeros([self.h, self.w])
        arr[:pad,:] = 1
        arr[:,:pad] = 1
        arr[-pad:,:] = 1
        arr[:,-pad:] = 1

        if drawcolor=="black":
            arr=1-arr

        return self.out(arr)
    
    def corner_squares(self, px=10, drawcolor="white"):
        """
        Generates squares on all corners

        Parameters:
            px = square size
            drawcolor = draw color as "white" or "black", the background will be the opposite
        """
        arr = np.zeros([self.h, self.w])
        arr[:px,:px] = 1
        arr[:px,-px:] = 1
        arr[-px:,-px:] = 1
        arr[-px:,:px] = 1

        if drawcolor=="black":
            arr=1-arr

        return self.out(arr)
    
    def slanted_squares(self, px, angle=4, sep=10, drawcolor="white", unprocessed=False):
        """
        Generates slanted squares

        Parameters:
            px = size of one square in pixels
            angle = tilt angle in degrees
            sep = separation between squares in pixels
            drawcolor = draw color as "white" or "black", the background will be the opposite
        """

    def slanted_edge(self, angle=4, offset=(0,0), dir="vert", color_ord="bw"):
        """
        Generates a slanted edge

        Parameters:
            angle = tilt angle in degrees
            color_ord = order as black-white "bw" or "wb"
        """

        angle = angle*2 # TODO: confirm angle? why times 2???

        
        y = np.arange(-self.h//2,self.h//2,1)+1-offset[0]
        x = np.arange(-self.w//2,self.w//2,1)+1-offset[1]
        X,Y = np.meshgrid(x,y)

        if dir == "vert":
            a = np.sin(np.radians(angle))*Y
            grid = X
        elif dir == "horz":
            a = np.sin(np.radians(angle))*X
            grid = Y

        if color_ord == "bw":
            grida = grid-a
        else:
            grida = a-grid

        grida[grida>0] = 1
        grida[grida<0] = 0
        
        return self.out(grida)

    def checkerboard(self, dx):
        # TODO
        # dx = checkerboard tile size in px
        
        arr = np.zeros(self.h,self.w)
        #arr[:n:,:n:] = 1
        return arr

    def generate_testchart(self, test_chart_type="sharpness"):
        if test_chart_type=="sharpness":
            img = self.line_grid2(interp=False) 
            img += self.crosshair(orientation=45, interp=False) 
            img += self.corner_squares(px=self.w//10) + self.circular_aperture()
            img += self.square_aperture(px=self.h//10, orientation=-4, offset=(-self.h//4, -self.w//4))
            img += self.square_aperture(px=self.h//10, orientation=-4, offset=(self.h//4, self.w//4))
            img += self.circular_aperture(rad=self.h//10, offset=(-self.h//4, self.w//4))
            img += self.circular_aperture(rad=self.h//10, offset=(self.h//4, -self.w//4))
            img += self.dot_grid(rad=self.h//200, sep=self.h//10)
        
        elif test_chart_type=="displacement":
            img = self.line_grid2(n_div=15)
            img += self.frame(pad=1)
            img += self.crosshair(orientation=45)

        elif test_chart_type=="test1":
            img = self.crosshair() + self.crosshair(orientation=45)
            img += self.dot_grid(rad=self.h//200, sep=self.h//10)

        return img

    def pad_and_position(self, img, offset=(0,0), orientation=0):
        """ 
        Takes an input image and zero pads it and positions according to the given offset

            Parameters:
                img = input image, 2D or 3D
                offset = (0,0) is the center
                orientation = TODO
        
        """

        im_h, im_w = img.shape[:2]
        im_c = 3 if len(img.shape) > 1 else 1

        arr = np.zeros([self.h, self.w, 3])  if im_c == 3 else np.zeros([self.h, self.w])

        y1 = self.h//2 + offset[0] - im_h//2
        y2 = self.h//2 + offset[0] + im_h//2
        x1 = self.w//2 + offset[1] - im_w//2
        x2 = self.w//2 + offset[1] + im_w//2
        arr[y1:y2,x1:x2,...] = img

        return arr.astype(self.ret_type)
    
    def get_test_sprite(self, angle=0, scale=1, offset=(0,0), bg="black"):

        def shear(arr, shear_amt, axis=0):
            arr = arr.astype(np.uint8)
            n = arr.shape[axis]
            if axis == 0:
                for i in range(n):
                    shift = round((i-n//2)*shear_amt)
                    arr[i,:] = np.roll(arr[i,:],shift,axis=0)
            elif axis == 1:
                for i in range(n):
                    shift = round((i-n//2)*shear_amt)
                    arr[:,i] = np.roll(arr[:,i],shift,axis=0)

            return arr

        def shear_rot(arr, angle, bg="black"):
            # Rotate an angle -90,90 with 3 shears
            org_dims = arr.shape
            angle = np.radians(angle)
            shear_x = -np.tan(angle/2)
            shear_y = np.sin(angle)
            bg_val = 0 if bg=="black" else 255
            arr = arr.astype(np.uint8)
            pad_size = (arr.shape[0]-1)//2
            tmp_arr = np.zeros([arr.shape[0]+pad_size*2, arr.shape[1]+pad_size*2, arr.shape[2]])
            for ch_i, ch in enumerate("RGB"):
                tmp_arr[:,:,ch_i] = np.pad(arr[:,:,ch_i],pad_size,'constant',constant_values=(bg_val))
            arr = tmp_arr
            tmp = arr.astype(np.uint8)
            
            tmp = shear(tmp, shear_x, axis=0)
            tmp = shear(tmp, shear_y, axis=1)
            tmp = shear(tmp, shear_x, axis=0)

            return crop(tmp, org_dims)
        
        sprite = np.ones((16, 16, 3), dtype=np.uint8) * 255 if bg=="white" else np.zeros((16,16,3))

        rng = np.random.default_rng(3)
        rng.shuffle(sprite, axis=0)

        # simple sliding window to create white borders for the sprite
        # borders = np.zeros(sprite.shape).astype(np.uint8)
        # mask = 1.0 * (sprite > 150)
        # mask = np.sum(mask, axis=2)
        # # print(np.max(mask))
        # row,col = sprite.shape[:2]
        # for r in range(1,row-1):
        #     for c in range(1,col-1):
        #         # print(np.sum(mask[r-1:r+1,c-1:c+1]))
        #         if mask[r,c] != 0:
        #             # if np.sum(mask[r-1:r+1,c-1:c+1]) > 9/4:
        #             #     borders[rr,cc] = np.mean()
        #             for rr in range(r-1,r+1):
        #                 for cc in range(c-1,c+1):
        #                     if mask[rr,cc] == 0:
        #                         borders[rr,cc] = np.sum(sprite[r-1:r+1,c-1:c+1])/8


        # sprite = borders + sprite
        # sprite = sprite.astype(np.uint8)
        # print(sprite.dtype)
        # print(sprite.shape)
        # print(np.max(sprite))

        # sprite = sprite.astype(np.uint8)

        # return sprite

        if angle:
            sprite = shear_rot(sprite, angle)

        if scale > 1:
            sprite = np.kron(sprite, np.ones((scale,scale,1))).astype(np.uint8)

        return self.pad_and_position(sprite, offset)