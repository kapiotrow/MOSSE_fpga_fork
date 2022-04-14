`timescale 1ns / 1ps
//-----------------------------------------------
// Company: agh
// Engineer: komorkiewicz
// Create Date: 23:14:48 04/19/2011 
// Description: vga generator
//-----------------------------------------------
module hdmi_in #(
    parameter IMG_PATH = ""
)
(
  //hdmi outputs
  output reg hdmi_clk,
  
  //master axis interface
  output reg [7:0] m_axis_0_tdata,
  output reg m_axis_0_tvalid,
  input m_axis_0_tready,
  output reg [17:0] pixel_cnt = 0
); 
//-----------------------------------------------
  parameter IGNORE_HEADER_CHARACTERS = 15;
  parameter height = 224; //resolution
  parameter width = 224; //resolution
  parameter total_pixels = height * width * 3;
//-----------------------------------------------
  reg skip_header = 1;
//  reg [17:0] pixel_cnt = 0;
//-----------------------------------------------
  
  //reg hdmi_clk=1'b0;
//-----------------------------------------------
initial
begin
  while(1)
  begin
    #1 hdmi_clk=1'b0;
	#1 hdmi_clk=1'b1;
  end
end  
//-----------------------------------------------
integer rgbfile,i,v,clo,cle,wl,x;

//-----------------------------------------------
//always @(posedge hdmi_clk)
//begin
//  hcounter<=hcounter+1;
  
//  eenab<=enab;

//  if(hcounter==(hr+hbp)) begin
//    hsync<=1'b0;
//  end
//  if(hcounter==(hr+hbp+hs)) begin
//    hsync<=1'b1;
//	 line<=1'b0;
//  end

//  if(hcounter<hr) 
//      h_enable<=1'b1;
//  else 
//		h_enable<=1'b0;
  
//  if(vcounter<vr) 
//		v_enable<=1'b1;
//  else 
//		v_enable<=1'b0;
		
//  if((v_enable==1'b1)&&(h_enable==1'b1))
//		enab<=1'b1;
//  else 
//		enab<=1'b0;
		  	  
//  if(hcounter==(hr+hbp+hs+hfp)) 
//  begin
//    hcounter<=0;
//	 line<=1'b1;
//  end
//end
//-----------------------------------------------
//TB only

always @(posedge hdmi_clk)
begin
    if(skip_header == 0 && m_axis_0_tready)
    begin
        m_axis_0_tdata <= $fgetc(rgbfile);
        m_axis_0_tvalid <= 1;
        pixel_cnt <= pixel_cnt + 1;
        
        if(pixel_cnt == total_pixels - 1)
        begin
            pixel_cnt <= 0;
            skip_header <= 1;
        end
    end
    
    else
    begin
        m_axis_0_tvalid <= 0;
    end
end


always @(posedge hdmi_clk)
begin
    if(skip_header == 1)
    begin  
        rgbfile = $fopen(IMG_PATH,"rb");
        
        // read header file
        for(i=0; i<IGNORE_HEADER_CHARACTERS; i=i+1)
        begin
            $fgetc(rgbfile); 
        end	
        skip_header <= 0;
    end
end

endmodule

